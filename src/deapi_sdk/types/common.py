from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from deapi_sdk._types import JobStatus


class JobSubmission(BaseModel):
    request_id: str


class PromptBoost(BaseModel):
    """What the inline prompt booster did to the prompt (present only once the boost ran)."""

    prompt: str | None = None
    prompt_original: str | None = None
    negative_prompt: str | None = None
    negative_prompt_original: str | None = None

    model_config = {"extra": "allow"}


class JobResult(BaseModel):
    status: JobStatus
    preview: str | None = None
    result_url: str | None = None
    results_alt_formats: dict[str, Any] | None = None
    result: str | None = None
    progress: float | None = 0.0
    # Failure surface (all null unless the job failed)
    error_code: str | None = None  # WORKER_TIMEOUT | WORKER_ERROR | PROCESSING_ERROR
    error_reason: str | None = None
    retryable: bool | None = None
    refunded: bool | None = None
    failed_at: str | None = None
    # Actual cost, e.g. {"amount": 0.01, "is_estimated": False}
    price: dict[str, Any] | float | None = None
    metadata: dict[str, Any] | None = None
    prompt_boosted: bool | None = None
    prompt_boost: PromptBoost | None = None
    reference_labels: Any = None  # not in the v2 spec (current responses omit it); kept for compatibility

    model_config = {"extra": "allow"}


class JobStatusEvent(BaseModel):
    """A real-time job status update delivered over WebSocket (``request.status.updated``).

    ``status`` is ``processing`` (repeated while ``progress`` grows) or ``done``;
    failed jobs are usually not announced over WebSocket. Unknown future statuses are kept
    as plain strings. ``progress`` is a percentage (0-100); the wire format sends
    it as a decimal string, which is converted to ``float``.
    """

    request_id: str
    # left_to_right: known values become JobStatus members instead of plain str
    status: JobStatus | str = Field(union_mode="left_to_right")
    progress: float | None = None
    preview: str | None = None
    result_url: str | None = None

    model_config = {"extra": "allow"}

    @field_validator("progress", mode="before")
    @classmethod
    def _parse_progress(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                return float(value)
            except ValueError:
                return None
        return value


class PriceResult(BaseModel):
    # The spec marks nothing required; ``price`` may be absent when ``estimate_basis`` is "unavailable".
    price: float | None = None
    is_estimated: bool | None = None
    # request_params | model_history | model_default | unavailable (only on estimated quotes)
    estimate_basis: str | None = None

    model_config = {"extra": "allow"}


class Balance(BaseModel):
    balance: float

    model_config = {"extra": "allow"}


class PaginationLinks(BaseModel):
    first: str | None = None
    last: str | None = None
    prev: str | None = None
    next: str | None = None

    model_config = {"extra": "allow"}


class PaginationMeta(BaseModel):
    current_page: int | None = None
    from_: int | None = Field(default=None, alias="from")
    last_page: int | None = None
    per_page: int | None = None
    to: int | None = None
    total: int | None = None
    path: str | None = None
    links: list[dict[str, Any]] | None = None

    model_config = {"extra": "allow", "populate_by_name": True}


class ModelInfo(BaseModel):
    name: str
    slug: str
    inference_types: list[str]
    status: str | None = None  # standard_model | test_model
    tags: list[str] = []
    info: dict[str, Any] | None = None
    loras: list[dict[str, Any]] | None = None
    languages: list[dict[str, Any]] | None = None

    model_config = {"extra": "allow"}

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, v: Any) -> Any:
        return [] if v is None else v

    def _info_section(self, key: str) -> dict[str, Any]:
        section = (self.info or {}).get(key)
        return section if isinstance(section, dict) else {}

    @property
    def limits(self) -> dict[str, Any]:
        """``info.limits`` (resolution, duration, steps, ...) or ``{}``."""
        return self._info_section("limits")

    @property
    def features(self) -> dict[str, Any]:
        """``info.features`` (feature flags such as ``supports_steps``) or ``{}``."""
        return self._info_section("features")

    @property
    def defaults(self) -> dict[str, Any]:
        """``info.defaults`` (default parameter values) or ``{}``."""
        return self._info_section("defaults")

    @field_validator("info", mode="before")
    @classmethod
    def _normalize_info(cls, v: Any) -> Any:
        # PHP/Laravel returns empty arrays as [] instead of {}
        if isinstance(v, list):
            return {} if len(v) == 0 else None
        return v


class ModelsResponse(BaseModel):
    data: list[ModelInfo]
    links: PaginationLinks | None = None
    meta: PaginationMeta | None = None

    model_config = {"extra": "allow"}

    @field_validator("links", "meta", mode="before")
    @classmethod
    def _lenient_pagination(cls, v: Any) -> Any:
        # PHP may send [] for empty objects; anything non-dict is dropped.
        return v if isinstance(v, dict) else None
