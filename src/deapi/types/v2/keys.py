from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator

from deapi.types.common import PaginationLinks, PaginationMeta


class ApiKey(BaseModel):
    """An API key as returned by ``GET /key``, ``GET /keys`` and ``GET /keys/{id}``. Never contains the secret."""

    id: int
    name: str | None = None
    preview: str | None = None
    role: str | None = None  # admin | api
    status: str | None = None  # active | revoked (disabled) | expired
    created_at: str | None = None
    expires_at: str | None = None
    last_used_at: str | None = None
    ip_whitelist: list[str] = []
    ip_whitelist_count: int | None = None
    limit: float | None = None
    reset_interval: str | None = None
    limit_used: float | None = None
    limit_remaining: float | None = None
    limit_resets_at: str | None = None

    model_config = {"extra": "allow"}

    @field_validator("ip_whitelist", mode="before")
    @classmethod
    def _norm_ip(cls, v: Any) -> Any:
        return [] if v is None else v


class ApiKeyCreated(ApiKey):
    """Response of ``POST /keys``. ``secret`` is shown ONCE; store it immediately."""

    secret: str | None = None


class ApiKeysPage(BaseModel):
    data: list[ApiKey] = []
    links: PaginationLinks | None = None
    meta: PaginationMeta | None = None

    model_config = {"extra": "allow"}

    @field_validator("links", "meta", mode="before")
    @classmethod
    def _lenient(cls, v: Any) -> Any:
        return v if isinstance(v, dict) else None
