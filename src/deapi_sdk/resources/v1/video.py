from __future__ import annotations

import json
from typing import Any

from deapi_sdk._cards import VIDEO_KEYS, afill_payload, fill_payload
from deapi_sdk._client import AsyncHTTPClient, SyncHTTPClient
from deapi_sdk._download import adownload_text, download_text
from deapi_sdk._exceptions import DeapiError
from deapi_sdk._files import FileInput, normalize_file
from deapi_sdk._payload import drop_none, form_bool, webhook_fields
from deapi_sdk._polling import AsyncJob, Job
from deapi_sdk.types.common import JobResult, PriceResult
from deapi_sdk.types.v1.video import VideoDescription, parse_video_description


class Video:
    """Sync video resource — generation, animation, audio-to-video, description, upscale (v1 and v2).

    ``describe`` / ``describe_price`` exist on v2 only (``videos/descriptions``).
    ``replace`` and ``remove_background`` are kept for compatibility: the routes still exist, but no model
    currently serves them (use ``animate`` with reference images/videos instead).
    """

    def __init__(self, client: SyncHTTPClient) -> None:
        self._client = client

    def generate(
        self,
        *,
        prompt: str,
        model: str,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        seed: int | None = None,
        frames: int | None = None,
        fps: int | None = None,
        negative_prompt: str | None = None,
        guidance: float | None = None,
        enhance_prompt: bool | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> Job:
        """Submit a text-to-video generation job.

        Unset ``width``, ``height``, ``frames`` and ``fps`` are taken from the model's ``info.defaults`` (one
        catalogue request per client, cached; disable with ``fill_model_defaults=False``); ``seed`` defaults to -1
        (random). Other parameters left as ``None`` are not sent.
        ``enhance_prompt`` rewrites the prompt server-side (v2; needs a prompt-booster guide for the model).
        """
        url = self._client._resolve_endpoint("txt2video")
        payload = _build_txt2video_payload(
            prompt=prompt, model=model, width=width, height=height,
            steps=steps, seed=seed, frames=frames, fps=fps,
            negative_prompt=negative_prompt, guidance=guidance,
            enhance_prompt=enhance_prompt,
            **webhook_fields(webhook_url, webhook_secret),
        )
        payload = fill_payload(self._client, payload, VIDEO_KEYS, seed=True)
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
        steps: int | None = None,
        seed: int | None = None,
        frames: int | None = None,
        fps: int | None = None,
        negative_prompt: str | None = None,
        guidance: float | None = None,
    ) -> PriceResult:
        """Calculate price for text-to-video generation.

        Unset ``width``, ``height``, ``frames`` and ``fps`` are taken from the model's
        ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``).
        """
        url = self._client._resolve_endpoint("txt2video_price")
        payload = _build_txt2video_payload(
            prompt=prompt, model=model, width=width, height=height,
            steps=steps, seed=seed, frames=frames, fps=fps,
            negative_prompt=negative_prompt, guidance=guidance,
        )
        payload = fill_payload(self._client, payload, VIDEO_KEYS)
        resp = self._client.post(url, json=payload)
        return PriceResult.model_validate(resp.get("data", resp))

    def animate(
        self,
        *,
        prompt: str,
        model: str,
        first_frame_image: FileInput | None = None,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        frames: int | None = None,
        fps: int | None = None,
        negative_prompt: str | None = None,
        last_frame_image: FileInput | None = None,
        guidance: float | None = None,
        steps: int | None = None,
        ref_images: list[FileInput] | None = None,
        ref_videos: list[FileInput] | None = None,
        ref_audios: list[FileInput] | None = None,
        enhance_prompt: bool | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> Job:
        """Submit an image-to-video animation job.

        Two input modes (pick one):

        * **Keyframes** — ``first_frame_image`` and, on models that support it, ``last_frame_image``.
        * **Full-reference** — ``ref_images`` and/or ``ref_videos``, optionally with ``ref_audios``
          (models whose card reports ``supports_image_refs`` / ``supports_video_refs`` /
          ``supports_audio_refs``). Cannot be combined with keyframes; audio alone is not enough.

        The order of each reference list binds the files to prompt tags: ``ref_images[0]`` is
        ``<Picture 1>``, ``ref_images[1]`` is ``<Picture 2>``, ``ref_videos[0]`` is ``<Video 1>`` and
        ``ref_audios[0]`` is ``<Audio 1>`` (``<Audio 2>`` if the reference video has its own audio track).
        Numbering is independent per type. Reference limits come from the model card
        (``max_ref_images``, ``max_ref_videos``, ``max_ref_audios``, ``max_ref_files``); the API never
        accepts more than 9 images, 1 video or 3 audios. Parts are sent as ``ref_images[0]``, ``ref_images[1]``, ...

        Unset ``width``, ``height``, ``frames`` and ``fps`` are taken from the model's
        ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``); ``seed`` defaults to -1 (random).

        Raises:
            ValueError: mixed modes, audio-only references, or no image source at all.
        """
        url = self._client._resolve_endpoint("img2video")
        data, files = _build_img2video_multipart(
            prompt=prompt, model=model, first_frame_image=first_frame_image,
            seed=seed, width=width, height=height, frames=frames, fps=fps,
            negative_prompt=negative_prompt, last_frame_image=last_frame_image,
            guidance=guidance, steps=steps,
            ref_images=ref_images, ref_videos=ref_videos, ref_audios=ref_audios,
            enhance_prompt=enhance_prompt,
            webhook=webhook_fields(webhook_url, webhook_secret),
        )
        data = fill_payload(self._client, data, VIDEO_KEYS, seed=True, form=True)
        resp = self._client.post(url, data=data, files=files)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return Job(request_id, self._client, status_url)

    def animate_price(
        self,
        *,
        model: str,
        width: int | None = None,
        height: int | None = None,
        frames: int | None = None,
        fps: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        ref_images_count: int | None = None,
        ref_videos_count: int | None = None,
        ref_audios_count: int | None = None,
    ) -> PriceResult:
        """Calculate price for image-to-video animation.

        For reference mode pass the *number* of reference files the job will carry, not the files
        (``ref_images_count`` 0-9, ``ref_videos_count`` 0-1, ``ref_audios_count`` 0-3).
        More than 2 reference images applies a x2 multiplier, any reference video x3 (the highest wins,
        they do not stack). Reference audio adds no surcharge. The quote does not check model limits.

        Unset ``width``, ``height``, ``frames`` and ``fps`` are taken from the model's
        ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``).
        """
        url = self._client._resolve_endpoint("img2video_price")
        payload = drop_none({
            "model": model, "width": width, "height": height,
            "frames": frames, "fps": fps, "steps": steps, "guidance": guidance,
            "ref_images_count": ref_images_count, "ref_videos_count": ref_videos_count,
            "ref_audios_count": ref_audios_count,
        })
        payload = fill_payload(self._client, payload, VIDEO_KEYS)
        resp = self._client.post(url, json=payload)
        return PriceResult.model_validate(resp.get("data", resp))

    def generate_from_audio(
        self,
        *,
        prompt: str,
        audio: FileInput,
        model: str,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        frames: int | None = None,
        fps: int | None = None,
        negative_prompt: str | None = None,
        first_frame_image: FileInput | None = None,
        last_frame_image: FileInput | None = None,
        guidance: float | None = None,
        steps: int | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> Job:
        """Submit an audio-to-video generation job (``videos/audio-syncs`` on v2).

        Unset ``width``, ``height``, ``frames`` and ``fps`` are taken from the model's ``info.defaults`` (one
        catalogue request per client, cached; disable with ``fill_model_defaults=False``); ``seed`` defaults to -1
        (random). Other parameters left as ``None`` are not sent.
        """
        url = self._client._resolve_endpoint("aud2video")
        data, files = _build_aud2video_multipart(
            prompt=prompt, audio=audio, model=model, width=width, height=height,
            seed=seed, frames=frames, fps=fps, negative_prompt=negative_prompt,
            first_frame_image=first_frame_image, last_frame_image=last_frame_image,
            guidance=guidance, steps=steps,
            webhook=webhook_fields(webhook_url, webhook_secret),
        )
        data = fill_payload(self._client, data, VIDEO_KEYS, seed=True, form=True)
        resp = self._client.post(url, data=data, files=files)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return Job(request_id, self._client, status_url)

    def generate_from_audio_price(
        self,
        *,
        model: str,
        width: int | None = None,
        height: int | None = None,
        frames: int | None = None,
        fps: int | None = None,
        guidance: float | None = None,
        steps: int | None = None,
    ) -> PriceResult:
        """Calculate price for audio-to-video generation.

        Unset ``width``, ``height``, ``frames`` and ``fps`` are taken from the model's
        ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``).
        """
        url = self._client._resolve_endpoint("aud2video_price")
        payload = drop_none({
            "model": model, "width": width, "height": height,
            "frames": frames, "fps": fps, "guidance": guidance, "steps": steps,
        })
        payload = fill_payload(self._client, payload, VIDEO_KEYS)
        resp = self._client.post(url, json=payload)
        return PriceResult.model_validate(resp.get("data", resp))

    def describe(
        self,
        *,
        model: str,
        video_url: str | None = None,
        video_file: FileInput | None = None,
        preset: str | None = None,
        include_metadata: bool | None = None,
        return_result_in_response: bool | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> Job:
        """Submit a video description job (v2, ``videos/descriptions``).

        Produces a timestamped index of what happens on screen (events, scenes, metadata); it reads the
        picture, not the audio (use ``audio.transcribe`` for speech). Pass exactly one source: ``video_url``
        (YouTube, Twitter/X, Twitch, Kick, TikTok) or ``video_file`` (file upload).

        ``preset`` is honoured only by models whose card has ``supports_presets`` (``available_presets``
        lists the choices: ``fast``, ``balanced``, ``detail``). ``include_metadata`` adds source metadata
        to the job status; only URL sources have any (a file returns null, uncharged).
        With ``return_result_in_response=True`` the JSON is also returned
        inline in ``JobResult.result``; otherwise download ``JobResult.result_url``. Use
        :meth:`get_description` to get a typed :class:`~deapi.types.v1.video.VideoDescription`.

        Raises:
            ValueError: not exactly one source.
        """
        url = self._client._resolve_endpoint("vid_description")
        data, files = _build_describe_form(
            model=model, video_url=video_url, video_file=video_file, preset=preset,
            include_metadata=include_metadata, return_result_in_response=return_result_in_response,
            webhook=webhook_fields(webhook_url, webhook_secret),
        )
        resp = self._client.post(url, data=data, files=files)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return Job(request_id, self._client, status_url)

    def describe_price(
        self,
        *,
        model: str,
        video_url: str | None = None,
        video_file: FileInput | None = None,
        duration_seconds: float | None = None,
        platform: str | None = None,
        preset: str | None = None,
        include_metadata: bool | None = None,
    ) -> PriceResult:
        """Calculate price for a video description.

        Pass exactly one source: ``video_url``, ``video_file`` or ``duration_seconds``. ``platform``
        (``youtube``, ``twitter``, ``twitch``, ``kick``, ``tiktok``) is only accepted with
        ``duration_seconds``; TikTok is priced on a fixed card by length (max 10 minutes) and a TikTok URL
        cannot be quoted, so quote it by duration. Use the same ``preset`` / ``include_metadata`` as the job.
        """
        url = self._client._resolve_endpoint("vid_description_price")
        data, files = _build_describe_price_form(
            model=model, video_url=video_url, video_file=video_file, duration_seconds=duration_seconds,
            platform=platform, preset=preset, include_metadata=include_metadata,
        )
        resp = self._client.post(url, data=data, files=files)
        return PriceResult.model_validate(resp.get("data", resp))

    def replace(
        self,
        *,
        video: FileInput,
        ref_image: FileInput,
        model: str,
        prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        seed: int | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> Job:
        """Submit a video replace job.

        Note: no model currently serves this route, so the API will reject it. To swap a character use
        :meth:`animate` with the character as ``ref_images`` and the source motion as ``ref_videos``.
        """
        url = self._client._resolve_endpoint("vid_replace")
        data, files = _build_vid_replace_multipart(
            video=video, ref_image=ref_image, model=model, prompt=prompt,
            width=width, height=height, steps=steps, seed=seed,
            webhook=webhook_fields(webhook_url, webhook_secret),
        )
        resp = self._client.post(url, data=data, files=files)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return Job(request_id, self._client, status_url)

    def replace_price(
        self,
        *,
        model: str,
        video: FileInput | None = None,
        duration: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> PriceResult:
        """Calculate price for video replace.

        Provide either ``video`` (file) or ``duration`` (seconds).
        If ``video`` is given without explicit dimensions, the server reads them from the file.
        No model currently serves this route (see :meth:`replace`).
        """
        if video is None and duration is None:
            raise ValueError("Either 'video' (file) or 'duration' (seconds) must be provided.")
        url = self._client._resolve_endpoint("vid_replace_price")
        if video is not None:
            data: dict[str, Any] = {"model": model}
            if width is not None:
                data["width"] = width
            if height is not None:
                data["height"] = height
            files = [("video", normalize_file(video, "video"))]
            resp = self._client.post(url, data=data, files=files)
        else:
            data_mp: dict[str, Any] = {"model": model}
            if duration is not None:
                data_mp["duration"] = str(duration)
            if width is not None:
                data_mp["width"] = str(width)
            if height is not None:
                data_mp["height"] = str(height)
            resp = self._client.post(url, data=data_mp)
        return PriceResult.model_validate(resp.get("data", resp))

    def upscale(
        self,
        *,
        video: FileInput,
        model: str,
        scale: int | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> Job:
        """Submit a video upscale job.

        ``scale`` is the upscale factor; only models whose card has a ``scale`` range accept it
        (e.g. FlashVSR: 2-4). Left unset, such models use their smallest factor.
        """
        url = self._client._resolve_endpoint("vid_upscale")
        data: dict[str, Any] = {"model": model}
        if scale is not None:
            data["scale"] = scale
        data.update(webhook_fields(webhook_url, webhook_secret))
        files = [("video", normalize_file(video, "video"))]
        resp = self._client.post(url, data=data, files=files)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return Job(request_id, self._client, status_url)

    def upscale_price(
        self,
        *,
        model: str,
        video: FileInput | None = None,
        width: int | None = None,
        height: int | None = None,
        scale: int | None = None,
        duration: float | None = None,
    ) -> PriceResult:
        """Calculate price for video upscale.

        Pass the ``video`` itself, or its ``width``, ``height`` and ``duration`` (seconds).
        """
        url = self._client._resolve_endpoint("vid_upscale_price")
        if video is not None:
            data: dict[str, Any] = {"model": model}
            if scale is not None:
                data["scale"] = scale
            if duration is not None:
                data["duration"] = duration
            files = [("video", normalize_file(video, "video"))]
            resp = self._client.post(url, data=data, files=files)
        else:
            payload = drop_none({
                "model": model, "width": width, "height": height, "scale": scale, "duration": duration,
            })
            resp = self._client.post(url, json=payload)
        return PriceResult.model_validate(resp.get("data", resp))

    def remove_background(
        self,
        *,
        video: FileInput,
        model: str,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> Job:
        """Submit a video background removal job.

        Note: no model currently serves this route, so the API will reject it.
        """
        url = self._client._resolve_endpoint("vid_rmbg")
        data: dict[str, Any] = {"model": model}
        data.update(webhook_fields(webhook_url, webhook_secret))
        files = [("video", normalize_file(video, "video"))]
        resp = self._client.post(url, data=data, files=files)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return Job(request_id, self._client, status_url)

    def remove_background_price(
        self,
        *,
        model: str,
        video: FileInput | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> PriceResult:
        """Calculate price for video background removal (no model currently serves this route)."""
        url = self._client._resolve_endpoint("vid_rmbg_price")
        if video is not None:
            data: dict[str, Any] = {"model": model}
            files = [("video", normalize_file(video, "video"))]
            resp = self._client.post(url, data=data, files=files)
        else:
            payload = drop_none({"model": model, "width": width, "height": height})
            resp = self._client.post(url, json=payload)
        return PriceResult.model_validate(resp.get("data", resp))

    def get_description(self, result: JobResult | str) -> VideoDescription:
        """Return the typed result of a finished :meth:`describe` job.

        ``result`` is the ``JobResult`` from ``job.wait()`` / ``job.status()``, a raw JSON string, or a
        ``result_url``. Inline ``result`` JSON is used when present; otherwise ``result_url`` is downloaded
        **without** the Authorization header (it is a storage URL; the API key is never sent to it).
        """
        inline, result_url = _description_source(result)
        if inline is not None:
            return parse_video_description(inline)
        return parse_video_description(_load_json(download_text(result_url, self._client._config.timeout)))


class AsyncVideo:
    """Async video resource — generation, animation, audio-to-video, description, upscale (v1 and v2).

    ``describe`` / ``describe_price`` exist on v2 only (``videos/descriptions``).
    ``replace`` and ``remove_background`` are kept for compatibility: the routes still exist, but no model
    currently serves them (use ``animate`` with reference images/videos instead).
    """

    def __init__(self, client: AsyncHTTPClient) -> None:
        self._client = client

    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        seed: int | None = None,
        frames: int | None = None,
        fps: int | None = None,
        negative_prompt: str | None = None,
        guidance: float | None = None,
        enhance_prompt: bool | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> AsyncJob:
        """Submit a text-to-video generation job.

        Unset ``width``, ``height``, ``frames`` and ``fps`` are taken from the model's ``info.defaults`` (one
        catalogue request per client, cached; disable with ``fill_model_defaults=False``); ``seed`` defaults to -1
        (random). Other parameters left as ``None`` are not sent.
        ``enhance_prompt`` rewrites the prompt server-side (v2; needs a prompt-booster guide for the model).
        """
        url = self._client._resolve_endpoint("txt2video")
        payload = _build_txt2video_payload(
            prompt=prompt, model=model, width=width, height=height,
            steps=steps, seed=seed, frames=frames, fps=fps,
            negative_prompt=negative_prompt, guidance=guidance,
            enhance_prompt=enhance_prompt,
            **webhook_fields(webhook_url, webhook_secret),
        )
        payload = await afill_payload(self._client, payload, VIDEO_KEYS, seed=True)
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
        steps: int | None = None,
        seed: int | None = None,
        frames: int | None = None,
        fps: int | None = None,
        negative_prompt: str | None = None,
        guidance: float | None = None,
    ) -> PriceResult:
        """Calculate price for text-to-video generation.

        Unset ``width``, ``height``, ``frames`` and ``fps`` are taken from the model's
        ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``).
        """
        url = self._client._resolve_endpoint("txt2video_price")
        payload = _build_txt2video_payload(
            prompt=prompt, model=model, width=width, height=height,
            steps=steps, seed=seed, frames=frames, fps=fps,
            negative_prompt=negative_prompt, guidance=guidance,
        )
        payload = await afill_payload(self._client, payload, VIDEO_KEYS)
        resp = await self._client.post(url, json=payload)
        return PriceResult.model_validate(resp.get("data", resp))

    async def animate(
        self,
        *,
        prompt: str,
        model: str,
        first_frame_image: FileInput | None = None,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        frames: int | None = None,
        fps: int | None = None,
        negative_prompt: str | None = None,
        last_frame_image: FileInput | None = None,
        guidance: float | None = None,
        steps: int | None = None,
        ref_images: list[FileInput] | None = None,
        ref_videos: list[FileInput] | None = None,
        ref_audios: list[FileInput] | None = None,
        enhance_prompt: bool | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> AsyncJob:
        """Submit an image-to-video animation job.

        Two input modes (pick one):

        * **Keyframes** — ``first_frame_image`` and, on models that support it, ``last_frame_image``.
        * **Full-reference** — ``ref_images`` and/or ``ref_videos``, optionally with ``ref_audios``
          (models whose card reports ``supports_image_refs`` / ``supports_video_refs`` /
          ``supports_audio_refs``). Cannot be combined with keyframes; audio alone is not enough.

        The order of each reference list binds the files to prompt tags: ``ref_images[0]`` is
        ``<Picture 1>``, ``ref_images[1]`` is ``<Picture 2>``, ``ref_videos[0]`` is ``<Video 1>`` and
        ``ref_audios[0]`` is ``<Audio 1>`` (``<Audio 2>`` if the reference video has its own audio track).
        Numbering is independent per type. Reference limits come from the model card
        (``max_ref_images``, ``max_ref_videos``, ``max_ref_audios``, ``max_ref_files``); the API never
        accepts more than 9 images, 1 video or 3 audios. Parts are sent as ``ref_images[0]``, ``ref_images[1]``, ...

        Unset ``width``, ``height``, ``frames`` and ``fps`` are taken from the model's
        ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``); ``seed`` defaults to -1 (random).

        Raises:
            ValueError: mixed modes, audio-only references, or no image source at all.
        """
        url = self._client._resolve_endpoint("img2video")
        data, files = _build_img2video_multipart(
            prompt=prompt, model=model, first_frame_image=first_frame_image,
            seed=seed, width=width, height=height, frames=frames, fps=fps,
            negative_prompt=negative_prompt, last_frame_image=last_frame_image,
            guidance=guidance, steps=steps,
            ref_images=ref_images, ref_videos=ref_videos, ref_audios=ref_audios,
            enhance_prompt=enhance_prompt,
            webhook=webhook_fields(webhook_url, webhook_secret),
        )
        data = await afill_payload(self._client, data, VIDEO_KEYS, seed=True, form=True)
        resp = await self._client.post(url, data=data, files=files)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return AsyncJob(request_id, self._client, status_url)

    async def animate_price(
        self,
        *,
        model: str,
        width: int | None = None,
        height: int | None = None,
        frames: int | None = None,
        fps: int | None = None,
        steps: int | None = None,
        guidance: float | None = None,
        ref_images_count: int | None = None,
        ref_videos_count: int | None = None,
        ref_audios_count: int | None = None,
    ) -> PriceResult:
        """Calculate price for image-to-video animation.

        For reference mode pass the *number* of reference files the job will carry, not the files
        (``ref_images_count`` 0-9, ``ref_videos_count`` 0-1, ``ref_audios_count`` 0-3).
        More than 2 reference images applies a x2 multiplier, any reference video x3 (the highest wins,
        they do not stack). Reference audio adds no surcharge. The quote does not check model limits.

        Unset ``width``, ``height``, ``frames`` and ``fps`` are taken from the model's
        ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``).
        """
        url = self._client._resolve_endpoint("img2video_price")
        payload = drop_none({
            "model": model, "width": width, "height": height,
            "frames": frames, "fps": fps, "steps": steps, "guidance": guidance,
            "ref_images_count": ref_images_count, "ref_videos_count": ref_videos_count,
            "ref_audios_count": ref_audios_count,
        })
        payload = await afill_payload(self._client, payload, VIDEO_KEYS)
        resp = await self._client.post(url, json=payload)
        return PriceResult.model_validate(resp.get("data", resp))

    async def generate_from_audio(
        self,
        *,
        prompt: str,
        audio: FileInput,
        model: str,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        frames: int | None = None,
        fps: int | None = None,
        negative_prompt: str | None = None,
        first_frame_image: FileInput | None = None,
        last_frame_image: FileInput | None = None,
        guidance: float | None = None,
        steps: int | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> AsyncJob:
        """Submit an audio-to-video generation job (``videos/audio-syncs`` on v2).

        Unset ``width``, ``height``, ``frames`` and ``fps`` are taken from the model's ``info.defaults`` (one
        catalogue request per client, cached; disable with ``fill_model_defaults=False``); ``seed`` defaults to -1
        (random). Other parameters left as ``None`` are not sent.
        """
        url = self._client._resolve_endpoint("aud2video")
        data, files = _build_aud2video_multipart(
            prompt=prompt, audio=audio, model=model, width=width, height=height,
            seed=seed, frames=frames, fps=fps, negative_prompt=negative_prompt,
            first_frame_image=first_frame_image, last_frame_image=last_frame_image,
            guidance=guidance, steps=steps,
            webhook=webhook_fields(webhook_url, webhook_secret),
        )
        data = await afill_payload(self._client, data, VIDEO_KEYS, seed=True, form=True)
        resp = await self._client.post(url, data=data, files=files)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return AsyncJob(request_id, self._client, status_url)

    async def generate_from_audio_price(
        self,
        *,
        model: str,
        width: int | None = None,
        height: int | None = None,
        frames: int | None = None,
        fps: int | None = None,
        guidance: float | None = None,
        steps: int | None = None,
    ) -> PriceResult:
        """Calculate price for audio-to-video generation.

        Unset ``width``, ``height``, ``frames`` and ``fps`` are taken from the model's
        ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``).
        """
        url = self._client._resolve_endpoint("aud2video_price")
        payload = drop_none({
            "model": model, "width": width, "height": height,
            "frames": frames, "fps": fps, "guidance": guidance, "steps": steps,
        })
        payload = await afill_payload(self._client, payload, VIDEO_KEYS)
        resp = await self._client.post(url, json=payload)
        return PriceResult.model_validate(resp.get("data", resp))

    async def describe(
        self,
        *,
        model: str,
        video_url: str | None = None,
        video_file: FileInput | None = None,
        preset: str | None = None,
        include_metadata: bool | None = None,
        return_result_in_response: bool | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> AsyncJob:
        """Submit a video description job (v2, ``videos/descriptions``).

        Produces a timestamped index of what happens on screen (events, scenes, metadata); it reads the
        picture, not the audio (use ``audio.transcribe`` for speech). Pass exactly one source: ``video_url``
        (YouTube, Twitter/X, Twitch, Kick, TikTok) or ``video_file`` (file upload).

        ``preset`` is honoured only by models whose card has ``supports_presets`` (``available_presets``
        lists the choices: ``fast``, ``balanced``, ``detail``). ``include_metadata`` adds source metadata
        to the job status; only URL sources have any (a file returns null, uncharged).
        With ``return_result_in_response=True`` the JSON is also returned
        inline in ``JobResult.result``; otherwise download ``JobResult.result_url``. Use
        :meth:`get_description` to get a typed :class:`~deapi.types.v1.video.VideoDescription`.

        Raises:
            ValueError: not exactly one source.
        """
        url = self._client._resolve_endpoint("vid_description")
        data, files = _build_describe_form(
            model=model, video_url=video_url, video_file=video_file, preset=preset,
            include_metadata=include_metadata, return_result_in_response=return_result_in_response,
            webhook=webhook_fields(webhook_url, webhook_secret),
        )
        resp = await self._client.post(url, data=data, files=files)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return AsyncJob(request_id, self._client, status_url)

    async def describe_price(
        self,
        *,
        model: str,
        video_url: str | None = None,
        video_file: FileInput | None = None,
        duration_seconds: float | None = None,
        platform: str | None = None,
        preset: str | None = None,
        include_metadata: bool | None = None,
    ) -> PriceResult:
        """Calculate price for a video description.

        Pass exactly one source: ``video_url``, ``video_file`` or ``duration_seconds``. ``platform``
        (``youtube``, ``twitter``, ``twitch``, ``kick``, ``tiktok``) is only accepted with
        ``duration_seconds``; TikTok is priced on a fixed card by length (max 10 minutes) and a TikTok URL
        cannot be quoted, so quote it by duration. Use the same ``preset`` / ``include_metadata`` as the job.
        """
        url = self._client._resolve_endpoint("vid_description_price")
        data, files = _build_describe_price_form(
            model=model, video_url=video_url, video_file=video_file, duration_seconds=duration_seconds,
            platform=platform, preset=preset, include_metadata=include_metadata,
        )
        resp = await self._client.post(url, data=data, files=files)
        return PriceResult.model_validate(resp.get("data", resp))

    async def replace(
        self,
        *,
        video: FileInput,
        ref_image: FileInput,
        model: str,
        prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        seed: int | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> AsyncJob:
        """Submit a video replace job.

        Note: no model currently serves this route, so the API will reject it. To swap a character use
        :meth:`animate` with the character as ``ref_images`` and the source motion as ``ref_videos``.
        """
        url = self._client._resolve_endpoint("vid_replace")
        data, files = _build_vid_replace_multipart(
            video=video, ref_image=ref_image, model=model, prompt=prompt,
            width=width, height=height, steps=steps, seed=seed,
            webhook=webhook_fields(webhook_url, webhook_secret),
        )
        resp = await self._client.post(url, data=data, files=files)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return AsyncJob(request_id, self._client, status_url)

    async def replace_price(
        self,
        *,
        model: str,
        video: FileInput | None = None,
        duration: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> PriceResult:
        """Calculate price for video replace.

        Provide either ``video`` (file) or ``duration`` (seconds).
        If ``video`` is given without explicit dimensions, the server reads them from the file.
        No model currently serves this route (see :meth:`replace`).
        """
        if video is None and duration is None:
            raise ValueError("Either 'video' (file) or 'duration' (seconds) must be provided.")
        url = self._client._resolve_endpoint("vid_replace_price")
        if video is not None:
            data: dict[str, Any] = {"model": model}
            if width is not None:
                data["width"] = width
            if height is not None:
                data["height"] = height
            files = [("video", normalize_file(video, "video"))]
            resp = await self._client.post(url, data=data, files=files)
        else:
            data_mp: dict[str, Any] = {"model": model}
            if duration is not None:
                data_mp["duration"] = str(duration)
            if width is not None:
                data_mp["width"] = str(width)
            if height is not None:
                data_mp["height"] = str(height)
            resp = await self._client.post(url, data=data_mp)
        return PriceResult.model_validate(resp.get("data", resp))

    async def upscale(
        self,
        *,
        video: FileInput,
        model: str,
        scale: int | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> AsyncJob:
        """Submit a video upscale job.

        ``scale`` is the upscale factor; only models whose card has a ``scale`` range accept it
        (e.g. FlashVSR: 2-4). Left unset, such models use their smallest factor.
        """
        url = self._client._resolve_endpoint("vid_upscale")
        data: dict[str, Any] = {"model": model}
        if scale is not None:
            data["scale"] = scale
        data.update(webhook_fields(webhook_url, webhook_secret))
        files = [("video", normalize_file(video, "video"))]
        resp = await self._client.post(url, data=data, files=files)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return AsyncJob(request_id, self._client, status_url)

    async def upscale_price(
        self,
        *,
        model: str,
        video: FileInput | None = None,
        width: int | None = None,
        height: int | None = None,
        scale: int | None = None,
        duration: float | None = None,
    ) -> PriceResult:
        """Calculate price for video upscale.

        Pass the ``video`` itself, or its ``width``, ``height`` and ``duration`` (seconds).
        """
        url = self._client._resolve_endpoint("vid_upscale_price")
        if video is not None:
            data: dict[str, Any] = {"model": model}
            if scale is not None:
                data["scale"] = scale
            if duration is not None:
                data["duration"] = duration
            files = [("video", normalize_file(video, "video"))]
            resp = await self._client.post(url, data=data, files=files)
        else:
            payload = drop_none({
                "model": model, "width": width, "height": height, "scale": scale, "duration": duration,
            })
            resp = await self._client.post(url, json=payload)
        return PriceResult.model_validate(resp.get("data", resp))

    async def remove_background(
        self,
        *,
        video: FileInput,
        model: str,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> AsyncJob:
        """Submit a video background removal job.

        Note: no model currently serves this route, so the API will reject it.
        """
        url = self._client._resolve_endpoint("vid_rmbg")
        data: dict[str, Any] = {"model": model}
        data.update(webhook_fields(webhook_url, webhook_secret))
        files = [("video", normalize_file(video, "video"))]
        resp = await self._client.post(url, data=data, files=files)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return AsyncJob(request_id, self._client, status_url)

    async def remove_background_price(
        self,
        *,
        model: str,
        video: FileInput | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> PriceResult:
        """Calculate price for video background removal (no model currently serves this route)."""
        url = self._client._resolve_endpoint("vid_rmbg_price")
        if video is not None:
            data: dict[str, Any] = {"model": model}
            files = [("video", normalize_file(video, "video"))]
            resp = await self._client.post(url, data=data, files=files)
        else:
            payload = drop_none({"model": model, "width": width, "height": height})
            resp = await self._client.post(url, json=payload)
        return PriceResult.model_validate(resp.get("data", resp))

    async def get_description(self, result: JobResult | str) -> VideoDescription:
        """Return the typed result of a finished :meth:`describe` job (see :meth:`Video.get_description`).

        The ``result_url`` download carries no Authorization header.
        """
        inline, result_url = _description_source(result)
        if inline is not None:
            return parse_video_description(inline)
        text = await adownload_text(result_url, self._client._config.timeout)
        return parse_video_description(_load_json(text))


def _as_list(items: Any) -> list[Any]:
    if items is None:
        return []
    if isinstance(items, (list, tuple)):
        return list(items)
    return [items]


def _build_txt2video_payload(
    *,
    prompt: str,
    model: str,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    seed: int | None = None,
    frames: int | None = None,
    fps: int | None = None,
    negative_prompt: str | None = None,
    guidance: float | None = None,
    enhance_prompt: bool | None = None,
    **extra: str,
) -> dict[str, Any]:
    payload = drop_none({
        "prompt": prompt,
        "model": model,
        "width": width,
        "height": height,
        "steps": steps,
        "seed": seed,
        "frames": frames,
        "fps": fps,
        "negative_prompt": negative_prompt,
        "guidance": guidance,
        "enhance_prompt": enhance_prompt,
    })
    payload.update(extra)
    return payload


def _build_img2video_multipart(
    *,
    prompt: str,
    model: str,
    first_frame_image: FileInput | None = None,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    frames: int | None = None,
    fps: int | None = None,
    negative_prompt: str | None = None,
    last_frame_image: FileInput | None = None,
    guidance: float | None = None,
    steps: int | None = None,
    ref_images: list[FileInput] | None = None,
    ref_videos: list[FileInput] | None = None,
    ref_audios: list[FileInput] | None = None,
    enhance_prompt: bool | None = None,
    webhook: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[tuple[str, tuple[str, bytes, str]]]]:
    """Build multipart form data and files list for img2video (keyframe or reference mode)."""
    images = _as_list(ref_images)
    videos = _as_list(ref_videos)
    audios = _as_list(ref_audios)
    has_refs = bool(images or videos or audios)

    if has_refs and (first_frame_image is not None or last_frame_image is not None):
        raise ValueError(
            "ref_images/ref_videos/ref_audios cannot be combined with first_frame_image/last_frame_image"
        )
    for label, items, cap in (("ref_images", images, 9), ("ref_videos", videos, 1), ("ref_audios", audios, 3)):
        if len(items) > cap:
            raise ValueError(f"{label} accepts at most {cap} files (got {len(items)})")
    if audios and not (images or videos):
        raise ValueError("ref_audios requires at least one ref_images or ref_videos entry")
    if first_frame_image is None and last_frame_image is None and not (images or videos):
        raise ValueError("Provide first_frame_image, or ref_images/ref_videos (reference mode)")

    data = drop_none({
        "prompt": prompt,
        "model": model,
        "seed": seed,
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "negative_prompt": negative_prompt,
        "guidance": guidance,
        "steps": steps,
    })
    if enhance_prompt is not None:
        data["enhance_prompt"] = form_bool(enhance_prompt)
    data.update(webhook or {})

    file_tuples: list[tuple[str, tuple[str, bytes, str]]] = []
    if first_frame_image is not None:
        file_tuples.append(("first_frame_image", normalize_file(first_frame_image, "first_frame_image")))
    if last_frame_image is not None:
        file_tuples.append(("last_frame_image", normalize_file(last_frame_image, "last_frame_image")))
    # Explicit indices (`ref_images[0]`, `ref_images[1]`, ...) bind files to <Picture N>/<Video N>/<Audio N>.
    for i, item in enumerate(images):
        file_tuples.append((f"ref_images[{i}]", normalize_file(item, "ref_image")))
    for i, item in enumerate(videos):
        file_tuples.append((f"ref_videos[{i}]", normalize_file(item, "ref_video")))
    for i, item in enumerate(audios):
        file_tuples.append((f"ref_audios[{i}]", normalize_file(item, "ref_audio")))

    return data, file_tuples


def _build_aud2video_multipart(
    *,
    prompt: str,
    audio: FileInput,
    model: str,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    frames: int | None = None,
    fps: int | None = None,
    negative_prompt: str | None = None,
    first_frame_image: FileInput | None = None,
    last_frame_image: FileInput | None = None,
    guidance: float | None = None,
    steps: int | None = None,
    webhook: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[tuple[str, tuple[str, bytes, str]]]]:
    """Build multipart form data and files list for aud2video."""
    data = drop_none({
        "prompt": prompt,
        "model": model,
        "width": width,
        "height": height,
        "seed": seed,
        "frames": frames,
        "fps": fps,
        "negative_prompt": negative_prompt,
        "guidance": guidance,
        "steps": steps,
    })
    data.update(webhook or {})

    file_tuples: list[tuple[str, tuple[str, bytes, str]]] = []
    file_tuples.append(("audio", normalize_file(audio, "audio")))
    if first_frame_image is not None:
        file_tuples.append(("first_frame_image", normalize_file(first_frame_image, "first_frame_image")))
    if last_frame_image is not None:
        file_tuples.append(("last_frame_image", normalize_file(last_frame_image, "last_frame_image")))

    return data, file_tuples


def _build_vid_replace_multipart(
    *,
    video: FileInput,
    ref_image: FileInput,
    model: str,
    prompt: str | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    seed: int | None = None,
    webhook: dict[str, str] | None = None,
) -> tuple[dict[str, Any], list[tuple[str, tuple[str, bytes, str]]]]:
    """Build multipart form data and files list for video replace."""
    data = drop_none({
        "model": model, "prompt": prompt, "width": width, "height": height, "steps": steps, "seed": seed,
    })
    data.update(webhook or {})

    file_tuples: list[tuple[str, tuple[str, bytes, str]]] = [
        ("video", normalize_file(video, "video")),
        ("ref_image", normalize_file(ref_image, "ref_image")),
    ]

    return data, file_tuples


def _build_describe_form(
    *,
    model: str,
    video_url: str | None,
    video_file: FileInput | None,
    preset: str | None,
    include_metadata: bool | None,
    return_result_in_response: bool | None,
    webhook: dict[str, str],
) -> tuple[dict[str, Any], list[tuple[str, tuple[str, bytes, str]]] | None]:
    if (video_url is None) == (video_file is None):
        raise ValueError("Provide exactly one of 'video_url' or 'video_file'")
    data: dict[str, Any] = {"model": model}
    if preset is not None:
        data["preset"] = preset
    if include_metadata is not None:
        data["include_metadata"] = form_bool(include_metadata)
    if return_result_in_response is not None:
        data["return_result_in_response"] = form_bool(return_result_in_response)
    data.update(webhook)
    if video_url is not None:
        data["video_url"] = video_url
        return data, None
    return data, [("video_file", normalize_file(video_file, "video_file"))]  # type: ignore[arg-type]


def _build_describe_price_form(
    *,
    model: str,
    video_url: str | None,
    video_file: FileInput | None,
    duration_seconds: float | None,
    platform: str | None,
    preset: str | None,
    include_metadata: bool | None,
) -> tuple[dict[str, Any], list[tuple[str, tuple[str, bytes, str]]] | None]:
    sources = [s for s in (video_url, video_file, duration_seconds) if s is not None]
    if len(sources) != 1:
        raise ValueError("Provide exactly one of 'video_url', 'video_file' or 'duration_seconds'")
    if platform is not None and duration_seconds is None:
        raise ValueError("platform is only accepted together with duration_seconds")
    data: dict[str, Any] = {"model": model}
    if preset is not None:
        data["preset"] = preset
    if include_metadata is not None:
        data["include_metadata"] = form_bool(include_metadata)
    if video_url is not None:
        data["video_url"] = video_url
        return data, None
    if duration_seconds is not None:
        data["duration_seconds"] = str(duration_seconds)
        if platform is not None:
            data["platform"] = platform
        return data, None
    return data, [("video_file", normalize_file(video_file, "video_file"))]  # type: ignore[arg-type]


def _description_source(result: JobResult | str) -> tuple[Any, str]:
    """Split a JobResult/str into (inline JSON or None, result_url)."""
    if isinstance(result, JobResult):
        if result.result:
            return result.result, ""
        if not result.result_url:
            raise ValueError("Job result has neither inline 'result' nor 'result_url' (is the job done?)")
        return None, result.result_url
    text = result.strip()
    if text.startswith("{"):
        return text, ""
    if not text.lower().startswith("https://"):
        raise ValueError("Expected a JobResult, a JSON string, or an https result_url")
    return None, text


def _load_json(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError as exc:
        raise DeapiError("Could not parse the video description result (invalid JSON)") from exc
