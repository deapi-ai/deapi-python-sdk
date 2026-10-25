from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator

from deapi.types.common import PaginationLinks, PaginationMeta


class Account(BaseModel):
    """Account billing (``GET /account``): balance, currency and automatic top-up settings."""

    balance: float | None = None
    currency: str | None = None
    auto_topup_enabled: bool | None = None
    auto_topup_threshold: float | None = None
    auto_topup_amount: float | None = None

    model_config = {"extra": "allow"}


class Payment(BaseModel):
    """One balance top-up (any status)."""

    id: Any = None
    status: str | None = None
    amount: float | None = None
    currency: str | None = None
    created_at: str | None = None

    model_config = {"extra": "allow"}


class PaymentsPage(BaseModel):
    data: list[Payment] = []
    links: PaginationLinks | None = None
    meta: PaginationMeta | None = None

    model_config = {"extra": "allow"}

    @field_validator("links", "meta", mode="before")
    @classmethod
    def _lenient(cls, v: Any) -> Any:
        return v if isinstance(v, dict) else None


class UsageReport(BaseModel):
    """Cost, requests and errors over a time window (account-wide or per key).

    The exact payload is passed through: unknown fields are kept as attributes, and when the API returns a
    bare list of groups it is exposed as ``groups``.
    """

    cost: float | None = None
    requests: int | None = None
    errors: int | None = None
    groups: list[dict[str, Any]] | None = None

    model_config = {"extra": "allow"}
