from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from deapi_sdk.types.common import PaginationLinks, PaginationMeta


class AutoTopUp(BaseModel):
    """Automatic top-up settings: top up ``amount`` when the balance falls to ``threshold``."""

    enabled: bool | None = None
    threshold: float | None = None
    amount: float | None = None  # null when no amount is configured

    model_config = {"extra": "allow"}


class Account(BaseModel):
    """Account billing (``GET /account``): balance, currency and the ``auto_top_up`` block."""

    balance: float | None = None
    currency: str | None = None
    auto_top_up: AutoTopUp | None = None

    model_config = {"extra": "allow"}

    @property
    def auto_topup_enabled(self) -> bool | None:
        return self.auto_top_up.enabled if self.auto_top_up else None

    @property
    def auto_topup_threshold(self) -> float | None:
        return self.auto_top_up.threshold if self.auto_top_up else None

    @property
    def auto_topup_amount(self) -> float | None:
        return self.auto_top_up.amount if self.auto_top_up else None


class Payment(BaseModel):
    """One balance top-up (any status)."""

    id: Any = None
    amount: float | None = None
    currency: str | None = None
    status: str | None = None  # pending | paid | disputed | cancelled | failed | expired
    provider: str | None = None
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


class UsageTotals(BaseModel):
    cost: float | None = None
    requests: int | None = None
    errors: int | None = None

    model_config = {"extra": "allow"}


class UsageGroup(BaseModel):
    """One bucket of a usage report. ``key`` is a date, type, model slug or key id depending on ``group_by``."""

    key: str | int | None = None
    name: str | None = None
    cost: float | None = None
    requests: int | None = None
    errors: int | None = None

    model_config = {"extra": "allow"}


class UsageReport(BaseModel):
    """Cost, requests and errors over a time window (account-wide or per key).

    ``totals`` holds the grand total and ``groups`` one entry per ``group_by`` bucket. ``cost``,
    ``requests`` and ``errors`` are shortcuts to ``totals``. The window is exposed as ``from_`` (the
    JSON field is ``from``) and ``to``.
    """

    from_: str | None = Field(default=None, alias="from")
    to: str | None = None
    group_by: str | None = None
    currency: str | None = None
    totals: UsageTotals | None = None
    groups: list[UsageGroup] | None = None

    model_config = {"extra": "allow", "populate_by_name": True}

    @property
    def cost(self) -> float | None:
        return self.totals.cost if self.totals else None

    @property
    def requests(self) -> int | None:
        return self.totals.requests if self.totals else None

    @property
    def errors(self) -> int | None:
        return self.totals.errors if self.totals else None
