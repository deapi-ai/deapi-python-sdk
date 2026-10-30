from __future__ import annotations

from typing import Any

from deapi_sdk._client import AsyncHTTPClient, SyncHTTPClient
from deapi_sdk._files import FileInput, normalize_file
from deapi_sdk.types.common import PriceResult
from deapi_sdk.types.prompts import EnhancePromptResult


class Prompts:
    """Sync prompt booster resource.

    All prompt boosting goes through one endpoint that takes a ``type``
    discriminator (e.g. ``"images.generations"``, ``"videos.animations"``,
    ``"audio.speech"``).
    """

    def __init__(self, client: SyncHTTPClient) -> None:
        self._client = client

    def enhance(
        self,
        *,
        prompt: str,
        type: str,
        model_slug: str,
        negative_prompt: str | None = None,
        image: FileInput | None = None,
    ) -> EnhancePromptResult:
        """Enhance a prompt via the unified v2 endpoint.

        ``type`` uses dot-notation matching the API resource (e.g.
        ``"images.generations"``, ``"images.edits"``, ``"videos.animations"``,
        ``"videos.generations"``, ``"audio.speech"``, ``"audio.music"``).
        ``image`` is required for ``images.edits`` and ``videos.animations``. Other types
        (upscales, background removal, OCR, transcription, embeddings) are rejected with 422;
        see ``deapi_sdk.types.prompts.PromptBoosterType``. The returned ``negative_prompt``
        is ignored by models without negative-prompt support.
        """
        url = self._client._resolve_endpoint("prompts_enhance")
        files = _build_enhance_parts(
            prompt=prompt, type=type, model_slug=model_slug,
            negative_prompt=negative_prompt, image=image,
        )
        resp = self._client.post(url, files=files)
        return EnhancePromptResult.model_validate(resp.get("data", resp))

    def enhance_price(
        self,
        *,
        prompt: str,
        type: str,
        model_slug: str,
        negative_prompt: str | None = None,
        image: FileInput | None = None,
    ) -> PriceResult:
        """Calculate price for prompt enhancement."""
        url = self._client._resolve_endpoint("prompts_enhance_price")
        files = _build_enhance_parts(
            prompt=prompt, type=type, model_slug=model_slug,
            negative_prompt=negative_prompt, image=image,
        )
        resp = self._client.post(url, files=files)
        return PriceResult.model_validate(resp.get("data", resp))


class AsyncPrompts:
    """Async prompt booster resource."""

    def __init__(self, client: AsyncHTTPClient) -> None:
        self._client = client

    async def enhance(
        self,
        *,
        prompt: str,
        type: str,
        model_slug: str,
        negative_prompt: str | None = None,
        image: FileInput | None = None,
    ) -> EnhancePromptResult:
        url = self._client._resolve_endpoint("prompts_enhance")
        files = _build_enhance_parts(
            prompt=prompt, type=type, model_slug=model_slug,
            negative_prompt=negative_prompt, image=image,
        )
        resp = await self._client.post(url, files=files)
        return EnhancePromptResult.model_validate(resp.get("data", resp))

    async def enhance_price(
        self,
        *,
        prompt: str,
        type: str,
        model_slug: str,
        negative_prompt: str | None = None,
        image: FileInput | None = None,
    ) -> PriceResult:
        url = self._client._resolve_endpoint("prompts_enhance_price")
        files = _build_enhance_parts(
            prompt=prompt, type=type, model_slug=model_slug,
            negative_prompt=negative_prompt, image=image,
        )
        resp = await self._client.post(url, files=files)
        return PriceResult.model_validate(resp.get("data", resp))


# --- Private helpers ---

def _build_enhance_parts(
    *,
    prompt: str,
    type: str,
    model_slug: str,
    negative_prompt: str | None = None,
    image: FileInput | None = None,
) -> list[tuple[str, Any]]:
    """Build the multipart form parts for the unified prompt enhancer.

    The v2 ``prompts/enhancements`` endpoint declares ``multipart/form-data``
    as its only content type in the OpenAPI spec, so we must force multipart
    encoding even when no image is sent. httpx triggers multipart encoding
    when ``files=`` is non-empty — passing text fields as ``(name, (None, value))``
    tuples in ``files=`` is the documented httpx pattern for mixing text parts
    with file parts in a single multipart request.
    """
    parts: list[tuple[str, Any]] = [
        ("prompt", (None, prompt)),
        ("type", (None, type)),
        ("model_slug", (None, model_slug)),
    ]
    if negative_prompt is not None:
        parts.append(("negative_prompt", (None, negative_prompt)))
    if image is not None:
        parts.append(("image", normalize_file(image, "image")))
    return parts
