from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from deapi_sdk._config import ClientConfig
from deapi_sdk._constants import ENDPOINTS
from deapi_sdk._exceptions import (
    AccountSuspendedError,
    ActiveKeyLimitReachedError,
    AuthenticationError,
    DeapiError,
    InsufficientBalanceError,
    InsufficientPermissionsError,
    IpNotAllowedError,
    KeyLimitExceededError,
    NetworkError,
    NotFoundError,
    PayloadTooLargeError,
    PaymentRequiredError,
    PermissionDeniedError,
    RateLimitError,
    ServerError,
    ValidationError,
)


class BaseClient:
    """Shared logic for sync and async HTTP clients."""

    def __init__(self, config: ClientConfig) -> None:
        self._config = config

    def _resolve_endpoint(self, operation: str) -> str:
        """Resolve an operation name to a full URL path."""
        version = self._config.api_version
        version_endpoints = ENDPOINTS.get(version)
        if version_endpoints is None:
            raise DeapiError(f"Unsupported API version: {version}")
        path = version_endpoints.get(operation)
        if path is None:
            raise DeapiError(f"Operation '{operation}' is not available in API {version}")
        return f"{self._config.base_url}{self._config.api_prefix}/{path}"

    def _default_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._config.api_key}",
            "Accept": "application/json",
            "User-Agent": self._config._user_agent,
        }

    @staticmethod
    def _parse_retry_after(value: str | None) -> float:
        """Parse a Retry-After header (seconds or HTTP-date) into seconds; garbage -> 0.0."""
        if not value:
            return 0.0
        value = value.strip()
        try:
            seconds = float(value)
        except ValueError:
            try:
                when = parsedate_to_datetime(value)
            except (TypeError, ValueError, IndexError):
                return 0.0
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max((when - datetime.now(timezone.utc)).total_seconds(), 0.0)
        if math.isnan(seconds) or math.isinf(seconds):
            return 0.0
        return max(seconds, 0.0)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Map HTTP error responses to typed exceptions."""
        if response.is_success:
            return

        status = response.status_code
        try:
            parsed = response.json()
        except Exception:
            parsed = None
        body: dict[str, Any] = parsed if isinstance(parsed, dict) else {}

        message: Any = body.get("message")
        if not isinstance(message, str) or not message:
            message = body.get("error")
        if not isinstance(message, str) or not message:
            message = response.text or f"HTTP {status}"

        raw_code = body.get("code")
        code = raw_code if isinstance(raw_code, str) and raw_code else None
        kwargs: dict[str, Any] = {"status_code": status, "body": body, "code": code}

        if status == 401:
            raise AuthenticationError(message, **kwargs)

        if status == 402:
            if code == "key_limit_exceeded":
                resets_at = body.get("resets_at")
                raise KeyLimitExceededError(
                    message,
                    **kwargs,
                    limit=self._to_float(body.get("limit")),
                    used=self._to_float(body.get("used")),
                    remaining=self._to_float(body.get("remaining")),
                    resets_at=resets_at if isinstance(resets_at, str) else None,
                )
            raise PaymentRequiredError(message, **kwargs)

        if status == 403:
            if code == "ip_not_allowed":
                raise IpNotAllowedError(message, **kwargs)
            if code == "insufficient_permissions":
                raise InsufficientPermissionsError(message, **kwargs)
            if code is None:
                # Only a code-less 403 means the account is suspended.
                raise AccountSuspendedError(message, **kwargs)
            raise PermissionDeniedError(message, **kwargs)

        if status == 404:
            raise NotFoundError(message, **kwargs)

        if status == 413:
            raise PayloadTooLargeError(message, **kwargs)

        if status == 422:
            errors = body.get("errors")
            if code == "insufficient_balance" or (isinstance(errors, dict) and "balance" in errors):
                raise InsufficientBalanceError(message, **kwargs, errors=errors)
            if code == "active_key_limit_reached":
                raise ActiveKeyLimitReachedError(message, **kwargs, errors=errors)
            raise ValidationError(message, **kwargs, errors=errors)

        if status == 429:
            raise RateLimitError(
                message,
                **kwargs,
                retry_after=self._parse_retry_after(response.headers.get("Retry-After")),
                limit_type=response.headers.get("X-RateLimit-Type", ""),
            )

        if status >= 500:
            raise ServerError(message, **kwargs)

        raise DeapiError(message, **kwargs)

    @staticmethod
    def _parse_success(response: httpx.Response) -> Any:
        """Parse a 2xx body: JSON, or ``{}`` when empty (e.g. 204)."""
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise DeapiError(
                f"Invalid JSON in response (HTTP {response.status_code})",
                status_code=response.status_code,
            ) from exc

    @staticmethod
    def _wrap_transport_error(exc: httpx.TransportError) -> NetworkError:
        err = NetworkError(str(exc))
        # Connection never established -> request provably never reached the server.
        err._never_sent = isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))  # type: ignore[attr-defined]
        return err

    def _should_retry(self, exc: Exception, attempt: int, method: str = "GET") -> float | None:
        """Determine if a request should be retried and return wait time in seconds.

        GET/HEAD/OPTIONS are retried on 429, 5xx and network errors. Other methods
        (POST/PATCH/DELETE) are retried only if the request provably did not reach
        processing: 429, or a connection that was never established. Retrying anything
        else could resubmit (and double-charge) a job the server already accepted.

        Returns None if no retry should happen.
        """
        if attempt >= self._config.max_retries:
            return None

        if isinstance(exc, RateLimitError):
            # Honour Retry-After, but never sleep longer than the configured cap:
            # a long or day-bucket value (seconds or a far-future HTTP-date) must
            # not park the request. The raw value stays on exc.retry_after.
            return min(max(exc.retry_after, 1.0), self._config.max_retry_after)

        safe = method.upper() in ("GET", "HEAD", "OPTIONS")

        if isinstance(exc, ServerError):
            return min(2.0 ** attempt, 8.0) if safe else None

        if isinstance(exc, NetworkError) and (safe or getattr(exc, "_never_sent", False)):
            return min(2.0 ** attempt, 8.0)

        return None


class SyncHTTPClient(BaseClient):
    """Synchronous HTTP client with retry logic.

    An ``httpx.Client`` may be injected so many per-user clients share one
    connection pool. Auth is sent per request (not baked into the transport),
    so a shared client serves every user with the right token. A caller-injected
    client is not closed by :meth:`close` unless ``owns_http_client=True``.
    """

    def __init__(
        self,
        config: ClientConfig,
        http_client: httpx.Client | None = None,
        *,
        owns_http_client: bool | None = None,
    ) -> None:
        super().__init__(config)
        self._headers = self._default_headers()
        if http_client is not None:
            self._http = http_client
            self._owns_http = bool(owns_http_client)
        else:
            self._http = httpx.Client(timeout=config.timeout)
            self._owns_http = True if owns_http_client is None else owns_http_client

    @property
    def owns_http_client(self) -> bool:
        return self._owns_http

    def close(self) -> None:
        # Only close a transport we created; never close one the caller injected.
        if self._owns_http:
            self._http.close()

    def request(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        last_exc: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                response = self._http.request(
                    method,
                    url,
                    json=json,
                    data=data,
                    files=files,
                    params=params,
                    headers=self._headers,
                )
                self._raise_for_status(response)
                return self._parse_success(response)
            except (RateLimitError, ServerError, NetworkError) as exc:
                last_exc = exc
                wait = self._should_retry(exc, attempt, method)
                if wait is not None:
                    time.sleep(wait)
                    continue
                raise
            except httpx.TransportError as exc:
                last_exc = self._wrap_transport_error(exc)
                wait = self._should_retry(last_exc, attempt, method)
                if wait is not None:
                    time.sleep(wait)
                    continue
                raise last_exc from exc

        raise last_exc  # type: ignore[misc]

    def get(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        return self.request("GET", url, params=params)

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    ) -> Any:
        return self.request("POST", url, json=json, data=data, files=files)

    def patch(self, url: str, *, json: dict[str, Any] | None = None) -> Any:
        return self.request("PATCH", url, json=json)

    def delete(self, url: str) -> Any:
        return self.request("DELETE", url)


class AsyncHTTPClient(BaseClient):
    """Asynchronous HTTP client with retry logic.

    An ``httpx.AsyncClient`` may be injected so many per-user clients share one
    connection pool. Auth is sent per request (not baked into the transport),
    so a shared client serves every user with the right token. A caller-injected
    client is not closed by :meth:`close` unless ``owns_http_client=True``.
    """

    def __init__(
        self,
        config: ClientConfig,
        http_client: httpx.AsyncClient | None = None,
        *,
        owns_http_client: bool | None = None,
    ) -> None:
        super().__init__(config)
        self._headers = self._default_headers()
        if http_client is not None:
            self._http = http_client
            self._owns_http = bool(owns_http_client)
        else:
            self._http = httpx.AsyncClient(timeout=config.timeout)
            self._owns_http = True if owns_http_client is None else owns_http_client

    @property
    def owns_http_client(self) -> bool:
        return self._owns_http

    async def close(self) -> None:
        # Only close a transport we created; never close one the caller injected.
        if self._owns_http:
            await self._http.aclose()

    async def request(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        last_exc: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                response = await self._http.request(
                    method,
                    url,
                    json=json,
                    data=data,
                    files=files,
                    params=params,
                    headers=self._headers,
                )
                self._raise_for_status(response)
                return self._parse_success(response)
            except (RateLimitError, ServerError, NetworkError) as exc:
                last_exc = exc
                wait = self._should_retry(exc, attempt, method)
                if wait is not None:
                    await asyncio.sleep(wait)
                    continue
                raise
            except httpx.TransportError as exc:
                last_exc = self._wrap_transport_error(exc)
                wait = self._should_retry(last_exc, attempt, method)
                if wait is not None:
                    await asyncio.sleep(wait)
                    continue
                raise last_exc from exc

        raise last_exc  # type: ignore[misc]

    async def get(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        return await self.request("GET", url, params=params)

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    ) -> Any:
        return await self.request("POST", url, json=json, data=data, files=files)

    async def patch(self, url: str, *, json: dict[str, Any] | None = None) -> Any:
        return await self.request("PATCH", url, json=json)

    async def delete(self, url: str) -> Any:
        return await self.request("DELETE", url)
