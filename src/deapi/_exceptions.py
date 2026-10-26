from __future__ import annotations

from typing import Any


class DeapiError(Exception):
    """Base exception for all SDK errors.

    ``code`` is the machine-readable error code the API sends next to ``message``
    (e.g. ``invalid_key``, ``ip_not_allowed``), or ``None`` when absent.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: dict | None = None,  # type: ignore[type-arg]
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body = body
        if code is None and isinstance(body, dict) and isinstance(body.get("code"), str):
            code = body["code"]
        self.code = code


class AuthenticationError(DeapiError):
    """401 — Missing, invalid, expired or revoked API key (see ``code``)."""


class PaymentRequiredError(DeapiError):
    """402 — Payment required (the API key's spend limit)."""


class KeyLimitExceededError(PaymentRequiredError):
    """402 ``key_limit_exceeded`` — the API key's spend limit is used up.

    This is the per-key spend limit, not the account balance (see ``InsufficientBalanceError``).
    ``resets_at`` is the raw value from the API (ISO datetime string or ``None``).
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = 402,
        body: dict | None = None,  # type: ignore[type-arg]
        code: str | None = None,
        limit: float | None = None,
        used: float | None = None,
        remaining: float | None = None,
        resets_at: str | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, body=body, code=code)
        self.limit = limit
        self.used = used
        self.remaining = remaining
        self.resets_at = resets_at


class PermissionDeniedError(DeapiError):
    """403 — The request is forbidden. Inspect ``code`` for the reason."""


class AccountSuspendedError(PermissionDeniedError):
    """403 without a code — Account suspended."""


class IpNotAllowedError(PermissionDeniedError):
    """403 ``ip_not_allowed`` — caller IP is outside the API key's IP allowlist."""


class InsufficientPermissionsError(PermissionDeniedError):
    """403 ``insufficient_permissions`` — the key's role cannot call this (admin-only) endpoint."""


class NotFoundError(DeapiError):
    """404 — Resource not found."""


class PayloadTooLargeError(DeapiError):
    """413 — Uploaded payload exceeds the size limit."""


def _normalize_errors(errors: Any) -> dict[str, list[str]]:
    if isinstance(errors, dict):
        return {
            str(k): [str(m) for m in v] if isinstance(v, (list, tuple)) else [str(v)]
            for k, v in errors.items()
        }
    if isinstance(errors, (list, tuple)) and errors:
        return {"_": [str(m) for m in errors]}
    return {}


class ValidationError(DeapiError):
    """422 — Request validation failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = 422,
        body: dict | None = None,  # type: ignore[type-arg]
        errors: Any = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, body=body, code=code)
        self.errors: dict[str, list[str]] = _normalize_errors(errors)


class InsufficientBalanceError(ValidationError):
    """422 ``insufficient_balance`` — not enough credits on the account."""


class ActiveKeyLimitReachedError(ValidationError):
    """422 ``active_key_limit_reached`` — would exceed the maximum number of active API keys."""


class RateLimitError(DeapiError):
    """429 — Rate limited."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = 429,
        body: dict | None = None,  # type: ignore[type-arg]
        retry_after: float = 0.0,
        limit_type: str = "",
        code: str | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, body=body, code=code)
        self.retry_after = retry_after
        self.limit_type = limit_type


class ServerError(DeapiError):
    """5xx — Server-side error."""


class JobTimeoutError(DeapiError):
    """Polling timeout — job didn't complete within max_wait."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        body: dict | None = None,  # type: ignore[type-arg]
        code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, body=body, code=code)
        self.request_id = request_id


class NetworkError(DeapiError):
    """Network connectivity error."""
