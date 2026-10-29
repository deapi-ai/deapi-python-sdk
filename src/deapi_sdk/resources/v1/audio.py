from __future__ import annotations

from typing import Any

from deapi_sdk._cards import MUSIC_KEYS, MUSIC_PRICE_KEYS, TTS_KEYS, afill_payload, fill_payload
from deapi_sdk._client import AsyncHTTPClient, SyncHTTPClient
from deapi_sdk._files import FileInput, normalize_file
from deapi_sdk._payload import drop_none, webhook_fields
from deapi_sdk._polling import AsyncJob, Job
from deapi_sdk.types.common import PriceResult


class Audio:
    """Sync audio resource -- TTS and music generation."""

    def __init__(self, client: SyncHTTPClient) -> None:
        self._client = client

    def synthesize(
        self,
        *,
        text: str,
        model: str,
        lang: str | None = None,
        format: str | None = None,
        speed: float | None = None,
        sample_rate: float | None = None,
        mode: str | None = None,
        voice: str | None = None,
        ref_audio: FileInput | None = None,
        ref_text: str | None = None,
        instruct: str | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> Job:
        """Submit a text-to-audio (TTS) job.

        Unset ``lang``, ``format``, ``speed`` and ``sample_rate`` are taken from the model's
        ``info.defaults`` (one catalogue request per client, cached; disable with ``fill_model_defaults=False``).
        ``ref_audio`` makes the request multipart.
        """
        url = self._client._resolve_endpoint("txt2audio")
        data, files = _build_tts_multipart(
            text=text, model=model, lang=lang, format=format, speed=speed,
            sample_rate=sample_rate, mode=mode, voice=voice, ref_audio=ref_audio,
            ref_text=ref_text, instruct=instruct,
            webhook_url=webhook_url, webhook_secret=webhook_secret,
        )
        data = fill_payload(self._client, data, TTS_KEYS, form=bool(files), tts=True)
        resp = self._client.post(url, data=data, files=files) if files else self._client.post(url, json=data)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return Job(request_id, self._client, status_url)

    def synthesize_price(
        self,
        *,
        model: str,
        lang: str | None = None,
        format: str | None = None,
        speed: float | None = None,
        sample_rate: float | None = None,
        text: str | None = None,
        count_text: int | None = None,
        mode: str | None = None,
        voice: str | None = None,
        instruct: str | None = None,
    ) -> PriceResult:
        """Calculate price for TTS (pass ``text`` or ``count_text``).

        Unset ``lang``, ``format``, ``speed`` and ``sample_rate`` are taken from the model's
        ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``).
        """
        url = self._client._resolve_endpoint("txt2audio_price")
        payload = drop_none({
            "model": model, "lang": lang, "format": format, "speed": speed,
            "sample_rate": sample_rate, "text": text, "count_text": count_text,
            "mode": mode, "voice": voice, "instruct": instruct,
        })
        payload = fill_payload(self._client, payload, TTS_KEYS, tts=True)
        resp = self._client.post(url, json=payload)
        return PriceResult.model_validate(resp.get("data", resp))

    def compose(
        self,
        *,
        caption: str,
        model: str,
        duration: float,
        inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        format: str | None = None,
        lyrics: str | None = None,
        bpm: int | None = None,
        keyscale: str | None = None,
        timesignature: int | None = None,
        vocal_language: str | None = None,
        reference_audio: FileInput | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> Job:
        """Submit a text-to-music generation job.

        ``reference_audio`` (style reference, length within the model's window) turns the
        request into multipart/form-data; without it the request is JSON. The API requires
        non-empty ``lyrics`` (use ``"[Instrumental]"`` for instrumentals) and an integer
        ``timesignature`` (2, 3, 4 or 6).

        Unset ``format``, ``inference_steps`` and ``guidance_scale`` are taken from the model's
        ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``); ``seed`` defaults to -1 (random).
        """
        url = self._client._resolve_endpoint("txt2music")
        payload, files = _build_music_request(
            caption=caption, model=model, duration=duration,
            inference_steps=inference_steps, guidance_scale=guidance_scale,
            seed=seed, format=format, lyrics=lyrics, bpm=bpm,
            keyscale=keyscale, timesignature=timesignature,
            vocal_language=vocal_language, reference_audio=reference_audio,
            webhook_url=webhook_url, webhook_secret=webhook_secret,
        )
        payload = fill_payload(self._client, payload, MUSIC_KEYS, seed=True, form=bool(files))
        resp = self._client.post(url, data=payload, files=files) if files else self._client.post(url, json=payload)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return Job(request_id, self._client, status_url)

    def compose_price(
        self,
        *,
        model: str,
        duration: float,
        inference_steps: int | None = None,
        guidance_scale: float | None = None,
        caption: str | None = None,
        lyrics: str | None = None,
        bpm: int | None = None,
        keyscale: str | None = None,
        timesignature: int | None = None,
        vocal_language: str | None = None,
    ) -> PriceResult:
        """Calculate price for music generation (only model, duration and steps affect the price).

        Unset ``inference_steps`` are taken from the model's ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``).
        """
        url = self._client._resolve_endpoint("txt2music_price")
        payload = drop_none({
            "model": model, "duration": duration, "inference_steps": inference_steps,
            "guidance_scale": guidance_scale, "caption": caption, "lyrics": lyrics,
            "bpm": bpm, "keyscale": keyscale, "timesignature": timesignature,
            "vocal_language": vocal_language,
        })
        payload = fill_payload(self._client, payload, MUSIC_PRICE_KEYS)
        resp = self._client.post(url, json=payload)
        return PriceResult.model_validate(resp.get("data", resp))


class AsyncAudio:
    """Async audio resource -- TTS and music generation."""

    def __init__(self, client: AsyncHTTPClient) -> None:
        self._client = client

    async def synthesize(
        self,
        *,
        text: str,
        model: str,
        lang: str | None = None,
        format: str | None = None,
        speed: float | None = None,
        sample_rate: float | None = None,
        mode: str | None = None,
        voice: str | None = None,
        ref_audio: FileInput | None = None,
        ref_text: str | None = None,
        instruct: str | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> AsyncJob:
        """Submit a text-to-audio (TTS) job.

        Unset ``lang``, ``format``, ``speed`` and ``sample_rate`` are taken from the model's
        ``info.defaults`` (one catalogue request per client, cached; disable with ``fill_model_defaults=False``).
        ``ref_audio`` makes the request multipart.
        """
        url = self._client._resolve_endpoint("txt2audio")
        data, files = _build_tts_multipart(
            text=text, model=model, lang=lang, format=format, speed=speed,
            sample_rate=sample_rate, mode=mode, voice=voice, ref_audio=ref_audio,
            ref_text=ref_text, instruct=instruct,
            webhook_url=webhook_url, webhook_secret=webhook_secret,
        )
        data = await afill_payload(self._client, data, TTS_KEYS, form=bool(files), tts=True)
        if files:
            resp = await self._client.post(url, data=data, files=files)
        else:
            resp = await self._client.post(url, json=data)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return AsyncJob(request_id, self._client, status_url)

    async def synthesize_price(
        self,
        *,
        model: str,
        lang: str | None = None,
        format: str | None = None,
        speed: float | None = None,
        sample_rate: float | None = None,
        text: str | None = None,
        count_text: int | None = None,
        mode: str | None = None,
        voice: str | None = None,
        instruct: str | None = None,
    ) -> PriceResult:
        """Calculate price for TTS (pass ``text`` or ``count_text``).

        Unset ``lang``, ``format``, ``speed`` and ``sample_rate`` are taken from the model's
        ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``).
        """
        url = self._client._resolve_endpoint("txt2audio_price")
        payload = drop_none({
            "model": model, "lang": lang, "format": format, "speed": speed,
            "sample_rate": sample_rate, "text": text, "count_text": count_text,
            "mode": mode, "voice": voice, "instruct": instruct,
        })
        payload = await afill_payload(self._client, payload, TTS_KEYS, tts=True)
        resp = await self._client.post(url, json=payload)
        return PriceResult.model_validate(resp.get("data", resp))

    async def compose(
        self,
        *,
        caption: str,
        model: str,
        duration: float,
        inference_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        format: str | None = None,
        lyrics: str | None = None,
        bpm: int | None = None,
        keyscale: str | None = None,
        timesignature: int | None = None,
        vocal_language: str | None = None,
        reference_audio: FileInput | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> AsyncJob:
        """Submit a text-to-music generation job.

        ``reference_audio`` (style reference, length within the model's window) turns the
        request into multipart/form-data; without it the request is JSON. The API requires
        non-empty ``lyrics`` (use ``"[Instrumental]"`` for instrumentals) and an integer
        ``timesignature`` (2, 3, 4 or 6).

        Unset ``format``, ``inference_steps`` and ``guidance_scale`` are taken from the model's
        ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``); ``seed`` defaults to -1 (random).
        """
        url = self._client._resolve_endpoint("txt2music")
        payload, files = _build_music_request(
            caption=caption, model=model, duration=duration,
            inference_steps=inference_steps, guidance_scale=guidance_scale,
            seed=seed, format=format, lyrics=lyrics, bpm=bpm,
            keyscale=keyscale, timesignature=timesignature,
            vocal_language=vocal_language, reference_audio=reference_audio,
            webhook_url=webhook_url, webhook_secret=webhook_secret,
        )
        payload = await afill_payload(self._client, payload, MUSIC_KEYS, seed=True, form=bool(files))
        if files:
            resp = await self._client.post(url, data=payload, files=files)
        else:
            resp = await self._client.post(url, json=payload)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return AsyncJob(request_id, self._client, status_url)

    async def compose_price(
        self,
        *,
        model: str,
        duration: float,
        inference_steps: int | None = None,
        guidance_scale: float | None = None,
        caption: str | None = None,
        lyrics: str | None = None,
        bpm: int | None = None,
        keyscale: str | None = None,
        timesignature: int | None = None,
        vocal_language: str | None = None,
    ) -> PriceResult:
        """Calculate price for music generation (only model, duration and steps affect the price).

        Unset ``inference_steps`` are taken from the model's ``info.defaults`` (one catalogue request per
        client, cached; disable with ``fill_model_defaults=False``).
        """
        url = self._client._resolve_endpoint("txt2music_price")
        payload = drop_none({
            "model": model, "duration": duration, "inference_steps": inference_steps,
            "guidance_scale": guidance_scale, "caption": caption, "lyrics": lyrics,
            "bpm": bpm, "keyscale": keyscale, "timesignature": timesignature,
            "vocal_language": vocal_language,
        })
        payload = await afill_payload(self._client, payload, MUSIC_PRICE_KEYS)
        resp = await self._client.post(url, json=payload)
        return PriceResult.model_validate(resp.get("data", resp))


# --- Private helpers ---

def _build_tts_multipart(
    *,
    text: str,
    model: str,
    lang: str | None = None,
    format: str | None = None,
    speed: float | None = None,
    sample_rate: float | None = None,
    mode: str | None = None,
    voice: str | None = None,
    ref_audio: FileInput | None = None,
    ref_text: str | None = None,
    instruct: str | None = None,
    webhook_url: str | None = None,
    webhook_secret: str | None = None,
) -> tuple[dict[str, Any], list[tuple[str, tuple[str, bytes, str]]]]:
    """Build form data for TTS. Returns (data_dict, file_tuples).

    When ref_audio is provided, the request must be multipart/form-data.
    Otherwise, the caller should send as JSON.
    """
    webhook = webhook_fields(webhook_url, webhook_secret)
    data = drop_none({
        "text": text, "model": model, "lang": lang, "format": format,
        "speed": speed, "sample_rate": sample_rate, "mode": mode, "voice": voice,
        "ref_text": ref_text, "instruct": instruct,
    })
    data.update(webhook)

    file_tuples: list[tuple[str, tuple[str, bytes, str]]] = []
    if ref_audio is not None:
        normalized = normalize_file(ref_audio, "ref_audio")
        file_tuples.append(("ref_audio", normalized))

    return data, file_tuples


def _build_music_request(
    *,
    caption: str,
    model: str,
    duration: float,
    inference_steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    format: str | None = None,
    lyrics: str | None = None,
    bpm: int | None = None,
    keyscale: str | None = None,
    timesignature: int | None = None,
    vocal_language: str | None = None,
    reference_audio: FileInput | None = None,
    webhook_url: str | None = None,
    webhook_secret: str | None = None,
) -> tuple[dict[str, Any], list[tuple[str, tuple[str, bytes, str]]]]:
    """Build the music request body. Returns (fields, files); files is non-empty -> multipart."""
    webhook = webhook_fields(webhook_url, webhook_secret)
    payload = drop_none({
        "caption": caption, "model": model, "duration": duration,
        "inference_steps": inference_steps, "guidance_scale": guidance_scale,
        "seed": seed, "format": format, "lyrics": lyrics, "bpm": bpm,
        "keyscale": keyscale, "timesignature": timesignature,
        "vocal_language": vocal_language,
    })
    payload.update(webhook)

    file_tuples: list[tuple[str, tuple[str, bytes, str]]] = []
    if reference_audio is not None:
        file_tuples.append(("reference_audio", normalize_file(reference_audio, "reference_audio")))
    return payload, file_tuples
