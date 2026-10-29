"""Helpers shared by resource payload builders."""

from __future__ import annotations

from typing import Any

# Mirrors the API's webhook rules:
#   webhook_url    nullable|url:https|max:2048|required_with:webhook_secret
#   webhook_secret nullable|string|min:32|max:255
WEBHOOK_URL_MAX_LENGTH = 2048
WEBHOOK_SECRET_MIN_LENGTH = 32
WEBHOOK_SECRET_MAX_LENGTH = 255


def webhook_fields(webhook_url: str | None, webhook_secret: str | None) -> dict[str, str]:
    """Return the webhook fields to add to a job payload, validated locally.

    The API checks the same rules, but failing here gives a clear message
    before any file is uploaded. Returns an empty dict when neither is set.

    Raises:
        ValueError: secret without URL, non-https URL, or bad lengths.
    """
    if not webhook_url and not webhook_secret:
        return {}
    if not webhook_url:
        raise ValueError("webhook_secret requires webhook_url")
    if not webhook_url.lower().startswith("https://"):
        raise ValueError("webhook_url must use https")
    if len(webhook_url) > WEBHOOK_URL_MAX_LENGTH:
        raise ValueError(f"webhook_url must be at most {WEBHOOK_URL_MAX_LENGTH} characters")

    fields = {"webhook_url": webhook_url}
    if webhook_secret:
        if not WEBHOOK_SECRET_MIN_LENGTH <= len(webhook_secret) <= WEBHOOK_SECRET_MAX_LENGTH:
            raise ValueError(
                f"webhook_secret must be {WEBHOOK_SECRET_MIN_LENGTH}-{WEBHOOK_SECRET_MAX_LENGTH} characters"
            )
        fields["webhook_secret"] = webhook_secret
    return fields


def drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of payload without None values (unset optional fields)."""
    return {k: v for k, v in payload.items() if v is not None}


def form_bool(value: bool) -> str:
    """Encode a bool for a multipart form field the way Laravel accepts it."""
    return "1" if value else "0"
