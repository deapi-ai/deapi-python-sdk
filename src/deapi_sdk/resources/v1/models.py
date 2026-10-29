from __future__ import annotations

import warnings
from typing import Any

from deapi_sdk._client import AsyncHTTPClient, SyncHTTPClient
from deapi_sdk.types.common import ModelInfo, ModelsResponse

# Server-side page-size cap for GET /models (``limit`` is capped at 50 by the API).
MAX_PAGE_SIZE = 50


def _build_params(
    limit: int | None,
    page: int,
    inference_types: list[str] | None,
    per_page: int | None,
    tags: list[str] | None = None,
    sort: str | None = None,
) -> dict[str, Any]:
    if per_page is not None:
        warnings.warn(
            "'per_page' is a deprecated alias of 'limit'; use 'limit' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        if limit is None:
            limit = per_page
    params: dict[str, Any] = {"page": page}
    if limit is not None:
        params["limit"] = limit
    if inference_types is not None:
        params["filter[inference_types]"] = ",".join(inference_types)
    if tags is not None:
        params["filter[tags]"] = ",".join(tags)
    if sort is not None:
        params["sort"] = sort
    return params


def _has_next(resp: ModelsResponse, page: int) -> bool:
    """``meta.last_page`` decides; without it, a full page means more may follow (never stop on a short one early)."""
    if not resp.data:
        return False
    last = resp.meta.last_page if resp.meta else None
    if last:
        return page < last
    return len(resp.data) >= MAX_PAGE_SIZE


def _find(models: list[ModelInfo], slug: str) -> ModelInfo | None:
    for m in models:
        if m.slug == slug:
            return m
    return None


class Models:
    """Sync models listing resource."""

    def __init__(self, client: SyncHTTPClient) -> None:
        self._client = client

    def list(
        self,
        *,
        limit: int | None = None,
        page: int = 1,
        inference_types: list[str] | None = None,
        per_page: int | None = None,
        tags: list[str] | None = None,
        sort: str | None = None,
    ) -> ModelsResponse:
        """List one page of available models with optional filtering.

        ``limit`` is the page size (default 50, server cap 50). ``per_page`` is a deprecated alias for ``limit``.
        ``tags`` filters by model tag (``new``, ``beta``, ...); ``sort`` is one of ``id``, ``name``,
        ``created_at``, ``updated_at``, optionally prefixed with ``-`` (default ``-created_at``).
        Check ``response.meta.last_page`` to know whether more pages exist, or use :meth:`list_all`.
        """
        url = self._client._resolve_endpoint("models")
        params = _build_params(limit, page, inference_types, per_page, tags, sort)
        resp = self._client.get(url, params=params)
        return ModelsResponse.model_validate(resp)

    def list_all(
        self, *, inference_types: list[str] | None = None, tags: list[str] | None = None, sort: str | None = None
    ) -> list[ModelInfo]:
        """Fetch every page (limit=50) and return all models."""
        models: list[ModelInfo] = []
        page = 1
        while True:
            resp = self.list(
                limit=MAX_PAGE_SIZE, page=page, inference_types=inference_types,
                tags=tags, sort=sort,
            )
            models.extend(resp.data)
            if not _has_next(resp, page):
                return models
            page += 1

    def get(self, slug: str) -> ModelInfo | None:
        """Find a model by slug in the full catalogue, or ``None`` if not available."""
        return _find(self.list_all(), slug)


class AsyncModels:
    """Async models listing resource."""

    def __init__(self, client: AsyncHTTPClient) -> None:
        self._client = client

    async def list(
        self,
        *,
        limit: int | None = None,
        page: int = 1,
        inference_types: list[str] | None = None,
        per_page: int | None = None,
        tags: list[str] | None = None,
        sort: str | None = None,
    ) -> ModelsResponse:
        """List one page of available models with optional filtering.

        ``limit`` is the page size (default 50, server cap 50). ``per_page`` is a deprecated alias for ``limit``.
        ``tags`` filters by model tag (``new``, ``beta``, ...); ``sort`` is one of ``id``, ``name``,
        ``created_at``, ``updated_at``, optionally prefixed with ``-`` (default ``-created_at``).
        """
        url = self._client._resolve_endpoint("models")
        params = _build_params(limit, page, inference_types, per_page, tags, sort)
        resp = await self._client.get(url, params=params)
        return ModelsResponse.model_validate(resp)

    async def list_all(
        self, *, inference_types: list[str] | None = None, tags: list[str] | None = None, sort: str | None = None
    ) -> list[ModelInfo]:
        """Fetch every page (limit=50) and return all models."""
        models: list[ModelInfo] = []
        page = 1
        while True:
            resp = await self.list(
                limit=MAX_PAGE_SIZE, page=page, inference_types=inference_types,
                tags=tags, sort=sort,
            )
            models.extend(resp.data)
            if not _has_next(resp, page):
                return models
            page += 1

    async def get(self, slug: str) -> ModelInfo | None:
        """Find a model by slug in the full catalogue, or ``None`` if not available."""
        return _find(await self.list_all(), slug)
