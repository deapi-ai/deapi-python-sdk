from __future__ import annotations

import warnings
from typing import Any

from deapi._client import AsyncHTTPClient, SyncHTTPClient
from deapi.types.common import ModelInfo, ModelsResponse

# Server-side page-size cap for GET /models (larger values are clamped, not rejected).
MAX_PAGE_SIZE = 50


def _build_params(
    limit: int | None,
    page: int,
    inference_types: list[str] | None,
    per_page: int | None,
) -> dict[str, Any]:
    if per_page is not None:
        warnings.warn(
            "'per_page' is deprecated and ignored by the API; use 'limit' instead.",
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
    return params


def _last_page(resp: ModelsResponse) -> int:
    return (resp.meta.last_page if resp.meta and resp.meta.last_page else None) or 1


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
    ) -> ModelsResponse:
        """List one page of available models with optional filtering.

        ``limit`` is the page size (server cap 50). ``per_page`` is a deprecated alias for ``limit``.
        Check ``response.meta.last_page`` to know whether more pages exist, or use :meth:`list_all`.
        """
        url = self._client._resolve_endpoint("models")
        params = _build_params(limit, page, inference_types, per_page)
        resp = self._client.get(url, params=params)
        return ModelsResponse.model_validate(resp)

    def list_all(self, *, inference_types: list[str] | None = None) -> list[ModelInfo]:
        """Fetch every page (limit=50) and return all models."""
        models: list[ModelInfo] = []
        page = 1
        while True:
            resp = self.list(limit=MAX_PAGE_SIZE, page=page, inference_types=inference_types)
            models.extend(resp.data)
            if page >= _last_page(resp) or not resp.data:
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
    ) -> ModelsResponse:
        """List one page of available models with optional filtering.

        ``limit`` is the page size (server cap 50). ``per_page`` is a deprecated alias for ``limit``.
        """
        url = self._client._resolve_endpoint("models")
        params = _build_params(limit, page, inference_types, per_page)
        resp = await self._client.get(url, params=params)
        return ModelsResponse.model_validate(resp)

    async def list_all(self, *, inference_types: list[str] | None = None) -> list[ModelInfo]:
        """Fetch every page (limit=50) and return all models."""
        models: list[ModelInfo] = []
        page = 1
        while True:
            resp = await self.list(limit=MAX_PAGE_SIZE, page=page, inference_types=inference_types)
            models.extend(resp.data)
            if page >= _last_page(resp) or not resp.data:
                return models
            page += 1

    async def get(self, slug: str) -> ModelInfo | None:
        """Find a model by slug in the full catalogue, or ``None`` if not available."""
        return _find(await self.list_all(), slug)
