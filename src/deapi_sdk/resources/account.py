from __future__ import annotations

from datetime import date, datetime
from typing import Any

from deapi_sdk._client import AsyncHTTPClient, SyncHTTPClient
from deapi_sdk.types.account import Account, PaymentsPage, UsageReport
from deapi_sdk.types.common import Balance


def _fmt_time(value: str | date | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def usage_params(
    from_: str | date | datetime | None,
    to: str | date | datetime | None,
    group_by: str | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    params: dict[str, Any] = dict(extra)
    if from_ is not None:
        params["from"] = _fmt_time(from_)
    if to is not None:
        params["to"] = _fmt_time(to)
    if group_by is not None:
        params["group_by"] = group_by
    return params


def page_params(
    limit: int | None, page: int | None, per_page: int | None, extra: dict[str, Any]
) -> dict[str, Any] | None:
    params: dict[str, Any] = dict(extra)
    if limit is not None:
        params["limit"] = limit
    if page is not None:
        params["page"] = page
    if per_page is not None:
        params["per_page"] = per_page
    return params or None


def unwrap(resp: Any) -> Any:
    return resp.get("data", resp) if isinstance(resp, dict) else resp


def parse_usage(resp: Any) -> UsageReport:
    data = unwrap(resp)
    if isinstance(data, list):  # defensive: a bare list of groups
        return UsageReport.model_validate({"groups": data})
    return UsageReport.model_validate(data)


class AccountResource:
    """Account billing and usage (API v2). Every call needs an ADMIN key (API keys get 403)."""

    def __init__(self, client: SyncHTTPClient) -> None:
        self._client = client

    def get(self) -> Account:
        """Balance, currency and auto top-up settings (``GET /account``). ADMIN key only."""
        resp = self._client.get(self._client._resolve_endpoint("account"))
        return Account.model_validate(unwrap(resp))

    def balance(self) -> Balance:
        """Current balance (``GET /account/balance``). ADMIN key only."""
        resp = self._client.get(self._client._resolve_endpoint("balance"))
        return Balance.model_validate(unwrap(resp))

    def usage(
        self,
        *,
        from_: str | date | datetime | None = None,
        to: str | date | datetime | None = None,
        group_by: str | None = None,
        **params: Any,
    ) -> UsageReport:
        """Usage of the whole account (``GET /account/usage``). ADMIN key only.

        ``group_by`` is one of ``day``, ``hour``, ``key``, ``type``, ``model``.
        """
        url = self._client._resolve_endpoint("account_usage")
        return parse_usage(self._client.get(url, params=usage_params(from_, to, group_by, params)))

    def payments(
        self,
        *,
        limit: int | None = None,
        page: int | None = None,
        per_page: int | None = None,
        **params: Any,
    ) -> PaymentsPage:
        """One page of balance top-ups (``GET /account/payments``). ADMIN key only.

        ``limit`` is the page size (default 50, max 100; ``per_page`` is an alias, ``limit`` wins).
        Items carry ``id``, ``amount``, ``currency``, ``status``, ``provider`` and ``created_at``.
        """
        resp = self._client.get(
            self._client._resolve_endpoint("account_payments"), params=page_params(limit, page, per_page, params)
        )
        return PaymentsPage.model_validate(resp)


class AsyncAccountResource:
    """Async variant of :class:`AccountResource`. Every call needs an ADMIN key."""

    def __init__(self, client: AsyncHTTPClient) -> None:
        self._client = client

    async def get(self) -> Account:
        """Balance, currency and auto top-up settings (``GET /account``). ADMIN key only."""
        resp = await self._client.get(self._client._resolve_endpoint("account"))
        return Account.model_validate(unwrap(resp))

    async def balance(self) -> Balance:
        """Current balance (``GET /account/balance``). ADMIN key only."""
        resp = await self._client.get(self._client._resolve_endpoint("balance"))
        return Balance.model_validate(unwrap(resp))

    async def usage(
        self,
        *,
        from_: str | date | datetime | None = None,
        to: str | date | datetime | None = None,
        group_by: str | None = None,
        **params: Any,
    ) -> UsageReport:
        """Usage of the whole account (``GET /account/usage``). ADMIN key only."""
        url = self._client._resolve_endpoint("account_usage")
        return parse_usage(await self._client.get(url, params=usage_params(from_, to, group_by, params)))

    async def payments(
        self,
        *,
        limit: int | None = None,
        page: int | None = None,
        per_page: int | None = None,
        **params: Any,
    ) -> PaymentsPage:
        """One page of balance top-ups (``GET /account/payments``). ADMIN key only. See the sync variant."""
        resp = await self._client.get(
            self._client._resolve_endpoint("account_payments"), params=page_params(limit, page, per_page, params)
        )
        return PaymentsPage.model_validate(resp)
