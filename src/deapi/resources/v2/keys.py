from __future__ import annotations

from datetime import date, datetime
from typing import Any

from deapi._client import AsyncHTTPClient, SyncHTTPClient
from deapi.resources.v2.account import parse_usage, unwrap, usage_params
from deapi.types.v2.account import UsageReport
from deapi.types.v2.keys import ApiKey, ApiKeyCreated, ApiKeysPage

# Sentinel: a field that was not passed must not be sent in a PATCH (None means "clear it").
_UNSET: Any = object()


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, (datetime, date)) else value


def _create_body(
    name: str,
    expires_at: Any,
    limit: float | None,
    reset_interval: str | None,
    ip_whitelist: list[str] | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {"name": name}
    for key, value in (
        ("expires_at", _iso(expires_at)),
        ("limit", limit),
        ("reset_interval", reset_interval),
        ("ip_whitelist", ip_whitelist),
    ):
        if value is not None:
            body[key] = value
    body.update(extra)
    return body


def _update_body(fields: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    body = {k: _iso(v) for k, v in fields.items() if v is not _UNSET}
    body.update(extra)
    return body


class KeysResource:
    """API key inspection and management (API v2).

    ``current()`` and ``current_usage()`` work with any key. Everything else needs an ADMIN key
    (API-role keys get 403 ``insufficient_permissions``).
    """

    def __init__(self, client: SyncHTTPClient) -> None:
        self._client = client

    def _key_url(self, key_id: int | str) -> str:
        return f"{self._client._resolve_endpoint('keys')}/{key_id}"

    def current(self) -> ApiKey:
        """The key sending the request (``GET /key``). Works with any key role."""
        return ApiKey.model_validate(unwrap(self._client.get(self._client._resolve_endpoint("key"))))

    def current_usage(
        self,
        *,
        from_: str | date | datetime | None = None,
        to: str | date | datetime | None = None,
        group_by: str | None = None,
        **params: Any,
    ) -> UsageReport:
        """Usage of the calling key (``GET /key/usage``). Works with any key role.

        ``group_by`` is one of ``day``, ``hour``, ``type``, ``model``.
        """
        url = self._client._resolve_endpoint("key_usage")
        return parse_usage(self._client.get(url, params=usage_params(from_, to, group_by, params)))

    def list(self, **params: Any) -> ApiKeysPage:
        """Every key of the account, newest first (``GET /keys``). ADMIN key only."""
        resp = self._client.get(self._client._resolve_endpoint("keys"), params=params or None)
        return ApiKeysPage.model_validate(resp)

    def get(self, key_id: int | str) -> ApiKey:
        """One key by id (``GET /keys/{id}``). ADMIN key only."""
        return ApiKey.model_validate(unwrap(self._client.get(self._key_url(key_id))))

    def usage(
        self,
        key_id: int | str,
        *,
        from_: str | date | datetime | None = None,
        to: str | date | datetime | None = None,
        group_by: str | None = None,
        **params: Any,
    ) -> UsageReport:
        """Usage of any key of the account (``GET /keys/{id}/usage``). ADMIN key only."""
        url = f"{self._key_url(key_id)}/usage"
        return parse_usage(self._client.get(url, params=usage_params(from_, to, group_by, params)))

    def create(
        self,
        *,
        name: str,
        expires_at: str | date | datetime | None = None,
        limit: float | None = None,
        reset_interval: str | None = None,
        ip_whitelist: list[str] | None = None,
        **extra: Any,
    ) -> ApiKeyCreated:
        """Create an API-role key (``POST /keys``). ADMIN key only.

        The returned ``secret`` is shown ONCE; store it immediately. ADMIN keys cannot be created over the API.
        """
        body = _create_body(name, expires_at, limit, reset_interval, ip_whitelist, extra)
        resp = self._client.request("POST", self._client._resolve_endpoint("keys"), json=body)
        return ApiKeyCreated.model_validate(unwrap(resp))

    def update(
        self,
        key_id: int | str,
        *,
        status: str = _UNSET,
        expires_at: str | date | datetime | None = _UNSET,
        limit: float | None = _UNSET,
        reset_interval: str | None = _UNSET,
        ip_whitelist: list[str] | None = _UNSET,
        **extra: Any,
    ) -> ApiKey:
        """Change a key (``PATCH /keys/{id}``). ADMIN key only.

        Only the fields you pass are sent; pass ``None`` to clear a field. Disable a key with
        ``status="revoked"`` and re-enable it with ``status="active"``.
        """
        body = _update_body(
            {
                "status": status,
                "expires_at": expires_at,
                "limit": limit,
                "reset_interval": reset_interval,
                "ip_whitelist": ip_whitelist,
            },
            extra,
        )
        resp = self._client.request("PATCH", self._key_url(key_id), json=body)
        return ApiKey.model_validate(unwrap(resp))

    def delete(self, key_id: int | str) -> None:
        """Permanently delete an API-role key (``DELETE /keys/{id}``). ADMIN key only."""
        self._client.request("DELETE", self._key_url(key_id))


class AsyncKeysResource:
    """Async variant of :class:`KeysResource`."""

    def __init__(self, client: AsyncHTTPClient) -> None:
        self._client = client

    def _key_url(self, key_id: int | str) -> str:
        return f"{self._client._resolve_endpoint('keys')}/{key_id}"

    async def current(self) -> ApiKey:
        """The key sending the request (``GET /key``). Works with any key role."""
        return ApiKey.model_validate(unwrap(await self._client.get(self._client._resolve_endpoint("key"))))

    async def current_usage(
        self,
        *,
        from_: str | date | datetime | None = None,
        to: str | date | datetime | None = None,
        group_by: str | None = None,
        **params: Any,
    ) -> UsageReport:
        """Usage of the calling key (``GET /key/usage``). Works with any key role."""
        url = self._client._resolve_endpoint("key_usage")
        return parse_usage(await self._client.get(url, params=usage_params(from_, to, group_by, params)))

    async def list(self, **params: Any) -> ApiKeysPage:
        """Every key of the account, newest first (``GET /keys``). ADMIN key only."""
        resp = await self._client.get(self._client._resolve_endpoint("keys"), params=params or None)
        return ApiKeysPage.model_validate(resp)

    async def get(self, key_id: int | str) -> ApiKey:
        """One key by id (``GET /keys/{id}``). ADMIN key only."""
        return ApiKey.model_validate(unwrap(await self._client.get(self._key_url(key_id))))

    async def usage(
        self,
        key_id: int | str,
        *,
        from_: str | date | datetime | None = None,
        to: str | date | datetime | None = None,
        group_by: str | None = None,
        **params: Any,
    ) -> UsageReport:
        """Usage of any key of the account (``GET /keys/{id}/usage``). ADMIN key only."""
        url = f"{self._key_url(key_id)}/usage"
        return parse_usage(await self._client.get(url, params=usage_params(from_, to, group_by, params)))

    async def create(
        self,
        *,
        name: str,
        expires_at: str | date | datetime | None = None,
        limit: float | None = None,
        reset_interval: str | None = None,
        ip_whitelist: list[str] | None = None,
        **extra: Any,
    ) -> ApiKeyCreated:
        """Create an API-role key (``POST /keys``). ADMIN key only. The ``secret`` is shown ONCE."""
        body = _create_body(name, expires_at, limit, reset_interval, ip_whitelist, extra)
        resp = await self._client.request("POST", self._client._resolve_endpoint("keys"), json=body)
        return ApiKeyCreated.model_validate(unwrap(resp))

    async def update(
        self,
        key_id: int | str,
        *,
        status: str = _UNSET,
        expires_at: str | date | datetime | None = _UNSET,
        limit: float | None = _UNSET,
        reset_interval: str | None = _UNSET,
        ip_whitelist: list[str] | None = _UNSET,
        **extra: Any,
    ) -> ApiKey:
        """Change a key (``PATCH /keys/{id}``). ADMIN key only.

        Only the fields you pass are sent; pass ``None`` to clear a field. Disable a key with
        ``status="revoked"`` and re-enable it with ``status="active"``.
        """
        body = _update_body(
            {
                "status": status,
                "expires_at": expires_at,
                "limit": limit,
                "reset_interval": reset_interval,
                "ip_whitelist": ip_whitelist,
            },
            extra,
        )
        resp = await self._client.request("PATCH", self._key_url(key_id), json=body)
        return ApiKey.model_validate(unwrap(resp))

    async def delete(self, key_id: int | str) -> None:
        """Permanently delete an API-role key (``DELETE /keys/{id}``). ADMIN key only."""
        await self._client.request("DELETE", self._key_url(key_id))
