from __future__ import annotations

from typing import Any, Union

from deapi._cards import IMAGE_KEYS, afill_payload, fill_payload
from deapi._client import AsyncHTTPClient, SyncHTTPClient
from deapi._files import FileInput, normalize_file, normalize_files
from deapi._payload import drop_none, form_bool, webhook_fields
from deapi._polling import AsyncJob, Job
from deapi.types.common import PriceResult
from deapi.types.v1.images import LoraWeight


class Images:
    """Sync image generation and transformation resource."""

    def __init__(self, client: SyncHTTPClient) -> None:
        self._client = client

    def generate(
        self,
        *,
        prompt: str,
        model: str,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        loras: list[Union[LoraWeight, dict[str, Any]]] | None = None,
        guidance: float | None = None,
        steps: int | None = None,
        quality: str | None = None,
        enhance_prompt: bool | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> Job:
        """Submit a text-to-image generation job.

        Unset ``width``, ``height`` and ``steps`` are taken from the model's ``info.defaults`` (one catalogue
        request per client, cached; disable with ``fill_model_defaults=False``) and ``seed`` defaults to -1
        (random). ``seed``, ``steps`` and ``guidance`` are otherwise only sent when set. ``quality`` is a model tier such as
        ``"low"``/``"medium"``/``"high"``. ``enhance_prompt=True`` boosts the prompt inside
        the job (v2; the boost fee is billed on the job).
        """
        url = self._client._resolve_endpoint("txt2img")
        payload = _build_txt2img_payload(
            prompt=prompt, model=model, width=width, height=height, seed=seed,
            negative_prompt=negative_prompt, loras=loras, guidance=guidance,
            steps=steps, quality=quality, enhance_prompt=enhance_prompt,
            webhook_url=webhook_url, webhook_secret=webhook_secret,
        )
        payload = fill_payload(self._client, payload, IMAGE_KEYS, seed=True)
        resp = self._client.post(url, json=payload)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return Job(request_id, self._client, status_url)

    def generate_price(
        self,
        *,
        prompt: str,
        model: str,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        loras: list[Union[LoraWeight, dict[str, Any]]] | None = None,
        guidance: float | None = None,
        steps: int | None = None,
        quality: str | None = None,
    ) -> PriceResult:
        """Calculate price for text-to-image generation.

        Unset ``width``, ``height`` and ``steps`` are taken from the model's ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``); ``seed`` defaults to -1 (random).
        """
        url = self._client._resolve_endpoint("txt2img_price")
        payload = _build_txt2img_payload(
            prompt=prompt, model=model, width=width, height=height, seed=seed,
            negative_prompt=negative_prompt, loras=loras, guidance=guidance,
            steps=steps, quality=quality,
        )
        payload = fill_payload(self._client, payload, IMAGE_KEYS, seed=True)
        resp = self._client.post(url, json=payload)
        return PriceResult.model_validate(resp.get("data", resp))

    def transform(
        self,
        *,
        prompt: str,
        model: str,
        steps: int | None = None,
        seed: int | None = None,
        image: FileInput | None = None,
        images: list[FileInput] | None = None,
        negative_prompt: str | None = None,
        loras: list[Union[LoraWeight, dict[str, Any]]] | None = None,
        width: int | None = None,
        height: int | None = None,
        guidance: float | None = None,
        quality: str | None = None,
        enhance_prompt: bool | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> Job:
        """Submit an image-to-image transformation (edit) job.

        Pass either ``image`` (one source) or ``images`` (several, sent as ``images[]``;
        the per-model maximum is ``max_input_images`` on the model card) -- not both.

        Unset ``width``, ``height`` and ``steps`` are taken from the model's ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``); ``seed`` defaults to -1 (random).
        """
        url = self._client._resolve_endpoint("img2img")
        data, files = _build_img2img_multipart(
            prompt=prompt, model=model, steps=steps, seed=seed,
            image=image, images=images, negative_prompt=negative_prompt,
            loras=loras, width=width, height=height, guidance=guidance,
            quality=quality, enhance_prompt=enhance_prompt,
            webhook_url=webhook_url, webhook_secret=webhook_secret,
        )
        data = fill_payload(self._client, data, IMAGE_KEYS, seed=True, form=True)
        resp = self._client.post(url, data=data, files=files)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return Job(request_id, self._client, status_url)

    def transform_price(
        self,
        *,
        prompt: str,
        model: str,
        steps: int | None = None,
        seed: int | None = None,
        image: FileInput | None = None,
        images: list[FileInput] | None = None,
        negative_prompt: str | None = None,
        loras: list[Union[LoraWeight, dict[str, Any]]] | None = None,
        width: int | None = None,
        height: int | None = None,
        guidance: float | None = None,
        quality: str | None = None,
    ) -> PriceResult:
        """Calculate price for an image edit.

        Some models price by the input size, so pass the same ``image``/``images`` the job
        will use. Without a source image the quote is sent as JSON.

        Unset ``width``, ``height`` and ``steps`` are taken from the model's ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``); ``seed`` defaults to -1 (random).
        """
        url = self._client._resolve_endpoint("img2img_price")
        json_body, data, files = _build_img2img_price_request(
            prompt=prompt, model=model, steps=steps, seed=seed, image=image, images=images,
            negative_prompt=negative_prompt, loras=loras, width=width, height=height,
            guidance=guidance, quality=quality,
        )
        fill_payload(self._client, json_body if json_body is not None else data, IMAGE_KEYS, seed=True, form=json_body is None)
        resp = self._client.post(url, json=json_body, data=data, files=files)
        return PriceResult.model_validate(resp.get("data", resp))

    def upscale(
        self,
        *,
        image: FileInput,
        model: str,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> Job:
        """Submit an image upscale job."""
        url = self._client._resolve_endpoint("img_upscale")
        data = {"model": model, **webhook_fields(webhook_url, webhook_secret)}
        files = [("image", normalize_file(image, "image"))]
        resp = self._client.post(url, data=data, files=files)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return Job(request_id, self._client, status_url)

    def upscale_price(
        self,
        *,
        model: str,
        image: FileInput | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> PriceResult:
        """Calculate price for image upscale (pass the image, or its width and height)."""
        url = self._client._resolve_endpoint("img_upscale_price")
        json_body, data, files = _build_size_price_request(model, image, width, height)
        resp = self._client.post(url, json=json_body, data=data, files=files)
        return PriceResult.model_validate(resp.get("data", resp))

    def remove_background(
        self,
        *,
        image: FileInput,
        model: str,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> Job:
        """Submit an image background removal job."""
        url = self._client._resolve_endpoint("img_rmbg")
        data = {"model": model, **webhook_fields(webhook_url, webhook_secret)}
        files = [("image", normalize_file(image, "image"))]
        resp = self._client.post(url, data=data, files=files)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return Job(request_id, self._client, status_url)

    def remove_background_price(
        self,
        *,
        model: str,
        image: FileInput | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> PriceResult:
        """Calculate price for image background removal (pass the image, or its width and height)."""
        url = self._client._resolve_endpoint("img_rmbg_price")
        json_body, data, files = _build_size_price_request(model, image, width, height)
        resp = self._client.post(url, json=json_body, data=data, files=files)
        return PriceResult.model_validate(resp.get("data", resp))


class AsyncImages:
    """Async image generation and transformation resource."""

    def __init__(self, client: AsyncHTTPClient) -> None:
        self._client = client

    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        loras: list[Union[LoraWeight, dict[str, Any]]] | None = None,
        guidance: float | None = None,
        steps: int | None = None,
        quality: str | None = None,
        enhance_prompt: bool | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> AsyncJob:
        """Submit a text-to-image generation job.

        Unset ``width``, ``height`` and ``steps`` are taken from the model's ``info.defaults`` (one catalogue
        request per client, cached; disable with ``fill_model_defaults=False``) and ``seed`` defaults to -1
        (random). ``seed``, ``steps`` and ``guidance`` are otherwise only sent when set. ``quality`` is a model tier such as
        ``"low"``/``"medium"``/``"high"``. ``enhance_prompt=True`` boosts the prompt inside
        the job (v2; the boost fee is billed on the job).
        """
        url = self._client._resolve_endpoint("txt2img")
        payload = _build_txt2img_payload(
            prompt=prompt, model=model, width=width, height=height, seed=seed,
            negative_prompt=negative_prompt, loras=loras, guidance=guidance,
            steps=steps, quality=quality, enhance_prompt=enhance_prompt,
            webhook_url=webhook_url, webhook_secret=webhook_secret,
        )
        payload = await afill_payload(self._client, payload, IMAGE_KEYS, seed=True)
        resp = await self._client.post(url, json=payload)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return AsyncJob(request_id, self._client, status_url)

    async def generate_price(
        self,
        *,
        prompt: str,
        model: str,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        loras: list[Union[LoraWeight, dict[str, Any]]] | None = None,
        guidance: float | None = None,
        steps: int | None = None,
        quality: str | None = None,
    ) -> PriceResult:
        """Calculate price for text-to-image generation.

        Unset ``width``, ``height`` and ``steps`` are taken from the model's ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``); ``seed`` defaults to -1 (random).
        """
        url = self._client._resolve_endpoint("txt2img_price")
        payload = _build_txt2img_payload(
            prompt=prompt, model=model, width=width, height=height, seed=seed,
            negative_prompt=negative_prompt, loras=loras, guidance=guidance,
            steps=steps, quality=quality,
        )
        payload = await afill_payload(self._client, payload, IMAGE_KEYS, seed=True)
        resp = await self._client.post(url, json=payload)
        return PriceResult.model_validate(resp.get("data", resp))

    async def transform(
        self,
        *,
        prompt: str,
        model: str,
        steps: int | None = None,
        seed: int | None = None,
        image: FileInput | None = None,
        images: list[FileInput] | None = None,
        negative_prompt: str | None = None,
        loras: list[Union[LoraWeight, dict[str, Any]]] | None = None,
        width: int | None = None,
        height: int | None = None,
        guidance: float | None = None,
        quality: str | None = None,
        enhance_prompt: bool | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> AsyncJob:
        """Submit an image-to-image transformation (edit) job.

        Pass either ``image`` (one source) or ``images`` (several, sent as ``images[]``;
        the per-model maximum is ``max_input_images`` on the model card) -- not both.

        Unset ``width``, ``height`` and ``steps`` are taken from the model's ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``); ``seed`` defaults to -1 (random).
        """
        url = self._client._resolve_endpoint("img2img")
        data, files = _build_img2img_multipart(
            prompt=prompt, model=model, steps=steps, seed=seed,
            image=image, images=images, negative_prompt=negative_prompt,
            loras=loras, width=width, height=height, guidance=guidance,
            quality=quality, enhance_prompt=enhance_prompt,
            webhook_url=webhook_url, webhook_secret=webhook_secret,
        )
        data = await afill_payload(self._client, data, IMAGE_KEYS, seed=True, form=True)
        resp = await self._client.post(url, data=data, files=files)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return AsyncJob(request_id, self._client, status_url)

    async def transform_price(
        self,
        *,
        prompt: str,
        model: str,
        steps: int | None = None,
        seed: int | None = None,
        image: FileInput | None = None,
        images: list[FileInput] | None = None,
        negative_prompt: str | None = None,
        loras: list[Union[LoraWeight, dict[str, Any]]] | None = None,
        width: int | None = None,
        height: int | None = None,
        guidance: float | None = None,
        quality: str | None = None,
    ) -> PriceResult:
        """Calculate price for an image edit.

        Some models price by the input size, so pass the same ``image``/``images`` the job
        will use. Without a source image the quote is sent as JSON.

        Unset ``width``, ``height`` and ``steps`` are taken from the model's ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``); ``seed`` defaults to -1 (random).
        """
        url = self._client._resolve_endpoint("img2img_price")
        json_body, data, files = _build_img2img_price_request(
            prompt=prompt, model=model, steps=steps, seed=seed, image=image, images=images,
            negative_prompt=negative_prompt, loras=loras, width=width, height=height,
            guidance=guidance, quality=quality,
        )
        await afill_payload(self._client, json_body if json_body is not None else data, IMAGE_KEYS, seed=True, form=json_body is None)
        resp = await self._client.post(url, json=json_body, data=data, files=files)
        return PriceResult.model_validate(resp.get("data", resp))

    async def upscale(
        self,
        *,
        image: FileInput,
        model: str,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> AsyncJob:
        """Submit an image upscale job."""
        url = self._client._resolve_endpoint("img_upscale")
        data = {"model": model, **webhook_fields(webhook_url, webhook_secret)}
        files = [("image", normalize_file(image, "image"))]
        resp = await self._client.post(url, data=data, files=files)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return AsyncJob(request_id, self._client, status_url)

    async def upscale_price(
        self,
        *,
        model: str,
        image: FileInput | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> PriceResult:
        """Calculate price for image upscale (pass the image, or its width and height)."""
        url = self._client._resolve_endpoint("img_upscale_price")
        json_body, data, files = _build_size_price_request(model, image, width, height)
        resp = await self._client.post(url, json=json_body, data=data, files=files)
        return PriceResult.model_validate(resp.get("data", resp))

    async def remove_background(
        self,
        *,
        image: FileInput,
        model: str,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> AsyncJob:
        """Submit an image background removal job."""
        url = self._client._resolve_endpoint("img_rmbg")
        data = {"model": model, **webhook_fields(webhook_url, webhook_secret)}
        files = [("image", normalize_file(image, "image"))]
        resp = await self._client.post(url, data=data, files=files)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return AsyncJob(request_id, self._client, status_url)

    async def remove_background_price(
        self,
        *,
        model: str,
        image: FileInput | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> PriceResult:
        """Calculate price for image background removal (pass the image, or its width and height)."""
        url = self._client._resolve_endpoint("img_rmbg_price")
        json_body, data, files = _build_size_price_request(model, image, width, height)
        resp = await self._client.post(url, json=json_body, data=data, files=files)
        return PriceResult.model_validate(resp.get("data", resp))


# --- Private helpers ---

_Files = list[tuple[str, tuple[str, bytes, str]]]


def _serialize_loras(
    loras: list[Union[LoraWeight, dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        lora.model_dump() if isinstance(lora, LoraWeight) else lora
        for lora in loras
    ]


def _build_txt2img_payload(
    *,
    prompt: str,
    model: str,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
    loras: list[Union[LoraWeight, dict[str, Any]]] | None = None,
    guidance: float | None = None,
    steps: int | None = None,
    quality: str | None = None,
    enhance_prompt: bool | None = None,
    webhook_url: str | None = None,
    webhook_secret: str | None = None,
) -> dict[str, Any]:
    payload = drop_none({
        "prompt": prompt,
        "model": model,
        "width": width,
        "height": height,
        "seed": seed,
        "negative_prompt": negative_prompt,
        "guidance": guidance,
        "steps": steps,
        "quality": quality,
        "enhance_prompt": enhance_prompt,
    })
    if loras is not None:
        payload["loras"] = _serialize_loras(loras)
    payload.update(webhook_fields(webhook_url, webhook_secret))
    return payload


def _check_image_args(image: FileInput | None, images: list[FileInput] | None) -> None:
    if image is not None and images is not None:
        raise ValueError("Provide either 'image' or 'images', not both.")


def _img2img_fields(
    *,
    prompt: str,
    model: str,
    steps: int | None,
    seed: int | None,
    negative_prompt: str | None,
    loras: list[Union[LoraWeight, dict[str, Any]]] | None,
    width: int | None,
    height: int | None,
    guidance: float | None,
    quality: str | None,
) -> dict[str, Any]:
    """Multipart text fields shared by edits and edits/price (loras as bracketed keys)."""
    data = drop_none({
        "prompt": prompt,
        "model": model,
        "steps": steps,
        "seed": seed,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "guidance": guidance,
        "quality": quality,
    })
    if loras is not None:
        for i, lora in enumerate(_serialize_loras(loras)):
            data[f"loras[{i}][name]"] = lora["name"]
            data[f"loras[{i}][weight]"] = lora["weight"]
    return data


def _img2img_files(image: FileInput | None, images: list[FileInput] | None) -> _Files:
    file_tuples: _Files = []
    if image is not None:
        file_tuples.append(("image", normalize_file(image, "image")))
    elif images is not None:
        for f in normalize_files(images, "image"):
            file_tuples.append(("images[]", f))
    return file_tuples


def _build_img2img_multipart(
    *,
    prompt: str,
    model: str,
    steps: int | None = None,
    seed: int | None = None,
    image: FileInput | None = None,
    images: list[FileInput] | None = None,
    negative_prompt: str | None = None,
    loras: list[Union[LoraWeight, dict[str, Any]]] | None = None,
    width: int | None = None,
    height: int | None = None,
    guidance: float | None = None,
    quality: str | None = None,
    enhance_prompt: bool | None = None,
    webhook_url: str | None = None,
    webhook_secret: str | None = None,
) -> tuple[dict[str, Any], _Files]:
    """Build multipart form data and files list for img2img."""
    _check_image_args(image, images)
    if image is None and images is None:
        raise ValueError("Either 'image' or 'images' is required.")
    webhook = webhook_fields(webhook_url, webhook_secret)

    data = _img2img_fields(
        prompt=prompt, model=model, steps=steps, seed=seed, negative_prompt=negative_prompt,
        loras=loras, width=width, height=height, guidance=guidance, quality=quality,
    )
    if enhance_prompt is not None:
        data["enhance_prompt"] = form_bool(enhance_prompt)
    data.update(webhook)
    return data, _img2img_files(image, images)


def _build_img2img_price_request(
    *,
    prompt: str,
    model: str,
    steps: int | None = None,
    seed: int | None = None,
    image: FileInput | None = None,
    images: list[FileInput] | None = None,
    negative_prompt: str | None = None,
    loras: list[Union[LoraWeight, dict[str, Any]]] | None = None,
    width: int | None = None,
    height: int | None = None,
    guidance: float | None = None,
    quality: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, _Files | None]:
    """Return (json, data, files) for edits/price: multipart when a source image is given."""
    _check_image_args(image, images)
    if image is None and images is None:
        payload = drop_none({
            "prompt": prompt, "model": model, "steps": steps, "seed": seed,
            "negative_prompt": negative_prompt, "width": width, "height": height,
            "guidance": guidance, "quality": quality,
        })
        if loras is not None:
            payload["loras"] = _serialize_loras(loras)
        return payload, None, None
    data = _img2img_fields(
        prompt=prompt, model=model, steps=steps, seed=seed, negative_prompt=negative_prompt,
        loras=loras, width=width, height=height, guidance=guidance, quality=quality,
    )
    return None, data, _img2img_files(image, images)


def _build_size_price_request(
    model: str, image: FileInput | None, width: int | None, height: int | None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, _Files | None]:
    """Return (json, data, files) for size-priced quotes: the image itself, or width/height."""
    if image is not None:
        return None, {"model": model}, [("image", normalize_file(image, "image"))]
    return drop_none({"model": model, "width": width, "height": height}), None, None
