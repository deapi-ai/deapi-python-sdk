from __future__ import annotations

from pydantic import BaseModel


class WebSocketConnection(BaseModel):
    """Where to open the realtime WebSocket (the ``connection`` block of ``GET /websocket``)."""

    protocol: str  # "pusher"
    key: str  # public application key of the WebSocket server (not a secret)
    host: str
    port: int
    tls: bool

    model_config = {"extra": "allow"}

    @property
    def url(self) -> str:
        """``wss://host`` / ``ws://host``; the port is included only when it is not the scheme default."""
        scheme = "wss" if self.tls else "ws"
        default_port = 443 if self.tls else 80
        host = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        suffix = "" if self.port == default_port else f":{self.port}"
        return f"{scheme}://{host}{suffix}"


class WebSocketConfig(BaseModel):
    """Realtime WebSocket parameters of the calling account (``GET /websocket``). No secrets."""

    client_id: str
    channel: str
    event: str
    auth_endpoint: str | None = None
    connection: WebSocketConnection

    model_config = {"extra": "allow"}
