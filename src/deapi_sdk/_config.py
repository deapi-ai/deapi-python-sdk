from __future__ import annotations

import os
from dataclasses import dataclass, field

from deapi_sdk._cards import parse_flag
from deapi_sdk._constants import PREFIXES
from deapi_sdk._version import __version__

_DEFAULT_BASE_URL = "https://api.deapi.ai"
_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_API_VERSION = "v2"
# Upper bound on how long we actually sleep for a 429 `Retry-After`, regardless of
# the value (or absolute HTTP-date) the server sends. Keeps a long/day-bucket
# Retry-After from parking a request for minutes/hours; the true server value is
# still carried on the raised RateLimitError for the caller to inspect.
_DEFAULT_MAX_RETRY_AFTER = 60.0


@dataclass(frozen=True)
class ClientConfig:
    api_key: str = field(repr=False)
    base_url: str = _DEFAULT_BASE_URL
    timeout: float = _DEFAULT_TIMEOUT
    max_retries: int = _DEFAULT_MAX_RETRIES
    api_version: str = _DEFAULT_API_VERSION
    fill_model_defaults: bool = True
    max_retry_after: float = _DEFAULT_MAX_RETRY_AFTER
    _user_agent: str = field(init=False, default=f"deapi-sdk/{__version__}")

    @classmethod
    def from_env(
        cls,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        api_version: str | None = None,
        fill_model_defaults: bool | None = None,
        max_retry_after: float | None = None,
    ) -> ClientConfig:
        resolved_key = (api_key or os.environ.get("DEAPI_API_KEY", "")).strip()
        if not resolved_key:
            raise ValueError(
                "API key is required. Pass api_key= or set the DEAPI_API_KEY environment variable."
            )
        return cls(
            api_key=resolved_key,
            base_url=base_url or os.environ.get("DEAPI_BASE_URL", _DEFAULT_BASE_URL),
            timeout=timeout if timeout is not None else _DEFAULT_TIMEOUT,
            max_retries=max_retries if max_retries is not None else _DEFAULT_MAX_RETRIES,
            api_version=api_version or os.environ.get("DEAPI_API_VERSION", _DEFAULT_API_VERSION),
            fill_model_defaults=(
                fill_model_defaults
                if fill_model_defaults is not None
                else parse_flag(os.environ.get("DEAPI_FILL_MODEL_DEFAULTS"))
            ),
            max_retry_after=(
                max_retry_after if max_retry_after is not None else _DEFAULT_MAX_RETRY_AFTER
            ),
        )

    @property
    def api_prefix(self) -> str:
        # v1: /api/v1/client, v2: /api/v2 (no /client/ segment).
        # Falls back to /api/{version}/client for unknown versions so a future
        # v3 with the same shape works without a code change here.
        return PREFIXES.get(self.api_version, f"/api/{self.api_version}/client")
