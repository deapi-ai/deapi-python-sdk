from __future__ import annotations

import warnings
from typing import Any

import httpx

from deapi._client import AsyncHTTPClient, SyncHTTPClient
from deapi._files import FileInput, normalize_file
from deapi._payload import drop_none, form_bool, webhook_fields
from deapi._polling import AsyncJob, Job
from deapi.transcript import Transcript, parse_transcript
from deapi.types.common import JobResult, PriceResult


_LEGACY_DEPRECATION = (
    "v2 has a unified transcription endpoint — call `transcription.create(...)` "
    "with `source_url` or `source_file`. Legacy method '{name}' is deprecated and "
    "will route through the unified endpoint automatically."
)


def _transcription_fields(
    *,
    model: str,
    include_ts: bool | None,
    ts_level: str | None,
    diarize: bool | None,
    lang: str | None,
    include_metadata: bool | None,
) -> dict[str, Any]:
    """Fields shared by the job and price payloads (None values dropped).

    ``ts_level`` and ``diarize`` only mean something with timestamps on (the API
    drops ``ts_level`` silently without them), so asking for either turns
    ``include_ts`` on unless the caller set it explicitly.
    """
    if include_ts is None and (ts_level or diarize):
        include_ts = True
    return drop_none({
        "model": model,
        "include_ts": include_ts,
        "ts_level": ts_level,
        "diarize": diarize,
        "lang": lang,
        "include_metadata": include_metadata,
    })


def _form(fields: dict[str, Any]) -> dict[str, Any]:
    """Encode bools for a multipart body."""
    return {k: form_bool(v) if isinstance(v, bool) else v for k, v in fields.items()}


def _check_source(source_url: str | None, source_file: FileInput | None) -> None:
    if source_url is not None and source_file is not None:
        raise ValueError("Provide either 'source_url' or 'source_file', not both.")
    if source_url is None and source_file is None:
        raise ValueError("Either 'source_url' or 'source_file' is required.")


def _build_create(
    *,
    source_url: str | None,
    source_file: FileInput | None,
    model: str,
    include_ts: bool | None,
    ts_level: str | None,
    diarize: bool | None,
    lang: str | None,
    include_metadata: bool | None,
    return_result_in_response: bool | None,
    webhook_url: str | None,
    webhook_secret: str | None,
) -> dict[str, Any]:
    """Build the keyword arguments for ``post()`` (JSON for URLs, multipart for files)."""
    _check_source(source_url, source_file)
    fields = _transcription_fields(
        model=model, include_ts=include_ts, ts_level=ts_level, diarize=diarize,
        lang=lang, include_metadata=include_metadata,
    )
    fields.update(drop_none({"return_result_in_response": return_result_in_response}))
    fields.update(webhook_fields(webhook_url, webhook_secret))
    if source_file is not None:
        return {"data": _form(fields), "files": [("source_file", normalize_file(source_file, "source_file"))]}
    return {"json": {"source_url": source_url, **fields}}


def _build_price(
    *,
    source_url: str | None,
    source_file: FileInput | None,
    model: str,
    include_ts: bool | None,
    ts_level: str | None,
    diarize: bool | None,
    lang: str | None,
    include_metadata: bool | None,
    platform: str | None,
    duration_seconds: float | None,
) -> dict[str, Any]:
    fields = _transcription_fields(
        model=model, include_ts=include_ts, ts_level=ts_level, diarize=diarize,
        lang=lang, include_metadata=include_metadata,
    )
    fields.update(drop_none({"platform": platform, "duration_seconds": duration_seconds}))
    if source_file is not None:
        return {"data": _form(fields), "files": [("source_file", normalize_file(source_file, "source_file"))]}
    if source_url is not None:
        fields["source_url"] = source_url
    return {"json": fields}


def _pick_result_source(result: JobResult, alt_format: str | None) -> tuple[str | None, str | None]:
    """Return ``(inline_text, url_to_download)`` for ``get_transcript``."""
    if alt_format is not None:
        alts = result.results_alt_formats or {}
        if alt_format not in alts:
            available = ", ".join(sorted(alts)) or "none"
            raise ValueError(f"No '{alt_format}' rendition on this job (available: {available}).")
        return None, alts[alt_format]
    if result.result is not None:
        return result.result, None
    if result.result_url:
        return None, result.result_url
    raise ValueError(
        "The job result has neither an inline 'result' nor a 'result_url'. "
        "Submit with return_result_in_response=True or wait until the job is done."
    )


def _format_hint(output_format: str | None, alt_format: str | None) -> str | None:
    # A txt rendition is plain text whatever the model's primary format is.
    if alt_format is not None and alt_format.lower() in ("txt", "text"):
        return "text"
    return output_format


class Transcription:
    """Sync transcription resource (v2).

    The v2 API has a single unified transcription endpoint. Legacy v1-style
    methods (``from_video_url``, ``from_audio_url``, ``from_audio_file``,
    ``from_video_file``) emit a ``DeprecationWarning`` and transparently
    route to ``create()``.

    The result of a finished job is either a JSON document or plain text,
    depending on the model; ``get_transcript()`` returns both as a ``Transcript``.
    """

    def __init__(self, client: SyncHTTPClient) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        include_ts: bool | None = None,
        ts_level: str | None = None,
        diarize: bool | None = None,
        lang: str | None = None,
        include_metadata: bool | None = None,
        source_url: str | None = None,
        source_file: FileInput | None = None,
        return_result_in_response: bool | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> Job:
        """Submit a transcription job via the unified endpoint.

        Provide either ``source_url`` (video/audio URL) or ``source_file``
        (audio/video file upload), not both.

        Args:
            model: Model slug, e.g. ``WhisperLargeV3Ct2``.
            include_ts: Return timestamps. Implied when ``ts_level`` or ``diarize`` is set.
            ts_level: ``segment``, ``word`` or ``char``; only levels the model lists
                under ``info.limits.timestamp_levels`` are accepted.
            diarize: Label speakers (models with ``supports_diarization``); implies timestamps.
            lang: ISO language code; unset or ``auto`` = detect.
            include_metadata: Return source info as ``metadata`` on the job status. Only URL sources have
                any (a file upload returns null and is never charged for the flag).
            webhook_secret: 32-255 char key to sign webhooks with; requires ``webhook_url``.
        """
        kwargs = _build_create(
            source_url=source_url, source_file=source_file, model=model, include_ts=include_ts,
            ts_level=ts_level, diarize=diarize, lang=lang, include_metadata=include_metadata,
            return_result_in_response=return_result_in_response,
            webhook_url=webhook_url, webhook_secret=webhook_secret,
        )
        resp = self._client.post(self._client._resolve_endpoint("transcribe"), **kwargs)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return Job(request_id, self._client, status_url)

    def create_price(
        self,
        *,
        model: str,
        include_ts: bool | None = None,
        ts_level: str | None = None,
        diarize: bool | None = None,
        lang: str | None = None,
        include_metadata: bool | None = None,
        platform: str | None = None,
        source_url: str | None = None,
        source_file: FileInput | None = None,
        duration_seconds: float | None = None,
    ) -> PriceResult:
        """Calculate the price of a transcription.

        Pass the same ``ts_level`` / ``diarize`` / ``lang`` / ``include_metadata`` you will pass
        to ``create()``: a quote that omits them comes in under the real charge.
        ``platform`` (e.g. ``tiktok``) matters only when quoting from ``duration_seconds``.
        """
        kwargs = _build_price(
            source_url=source_url, source_file=source_file, model=model, include_ts=include_ts,
            ts_level=ts_level, diarize=diarize, lang=lang, include_metadata=include_metadata,
            platform=platform, duration_seconds=duration_seconds,
        )
        resp = self._client.post(self._client._resolve_endpoint("transcribe_price"), **kwargs)
        return PriceResult.model_validate(resp.get("data", resp))

    def get_transcript(
        self,
        result: JobResult,
        *,
        output_format: str | None = None,
        alt_format: str | None = None,
    ) -> Transcript:
        """Turn a finished job's result into a ``Transcript``.

        Uses the inline ``result`` when present, otherwise downloads ``result_url``.
        The download is a plain GET that carries **no** Authorization header (result
        URLs are storage URLs; the API key must not leak to them).

        Args:
            result: ``job.wait()`` / ``job.status()`` result.
            output_format: The model card's ``info.limits.output_format`` (``"json"``);
                ``"text"`` forces plain text. Auto-detected when omitted.
            alt_format: Pick a rendition from ``results_alt_formats`` (e.g. ``"txt"``)
                instead of the primary result.
        """
        inline, url = _pick_result_source(result, alt_format)
        if inline is None:
            resp = httpx.get(url, timeout=self._client._config.timeout, follow_redirects=True)  # type: ignore[arg-type]
            self._client._raise_for_status(resp)
            inline = resp.text
        return parse_transcript(inline, output_format=_format_hint(output_format, alt_format))

    # --- Legacy methods (deprecated, route to create) ---

    def from_video_url(
        self,
        *,
        video_url: str,
        include_ts: bool,
        model: str,
        return_result_in_response: bool | None = None,
        webhook_url: str | None = None,
        **options: Any,
    ) -> Job:
        warnings.warn(
            _LEGACY_DEPRECATION.format(name="from_video_url"),
            DeprecationWarning,
            stacklevel=2,
        )
        return self.create(
            source_url=video_url,
            include_ts=include_ts,
            model=model,
            return_result_in_response=return_result_in_response,
            webhook_url=webhook_url,
            **options,
        )

    def from_video_url_price(
        self,
        *,
        include_ts: bool,
        model: str,
        video_url: str | None = None,
        duration_seconds: float | None = None,
        **options: Any,
    ) -> PriceResult:
        warnings.warn(
            _LEGACY_DEPRECATION.format(name="from_video_url_price"),
            DeprecationWarning,
            stacklevel=2,
        )
        return self.create_price(
            include_ts=include_ts,
            model=model,
            source_url=video_url,
            duration_seconds=duration_seconds,
            **options,
        )

    def from_audio_url(
        self,
        *,
        audio_url: str,
        include_ts: bool,
        model: str,
        return_result_in_response: bool | None = None,
        webhook_url: str | None = None,
        **options: Any,
    ) -> Job:
        warnings.warn(
            _LEGACY_DEPRECATION.format(name="from_audio_url"),
            DeprecationWarning,
            stacklevel=2,
        )
        return self.create(
            source_url=audio_url,
            include_ts=include_ts,
            model=model,
            return_result_in_response=return_result_in_response,
            webhook_url=webhook_url,
            **options,
        )

    def from_audio_url_price(
        self,
        *,
        include_ts: bool,
        model: str,
        audio_url: str | None = None,
        duration_seconds: float | None = None,
        **options: Any,
    ) -> PriceResult:
        warnings.warn(
            _LEGACY_DEPRECATION.format(name="from_audio_url_price"),
            DeprecationWarning,
            stacklevel=2,
        )
        return self.create_price(
            include_ts=include_ts,
            model=model,
            source_url=audio_url,
            duration_seconds=duration_seconds,
            **options,
        )

    def from_audio_file(
        self,
        *,
        audio: FileInput,
        include_ts: bool,
        model: str,
        return_result_in_response: bool | None = None,
        webhook_url: str | None = None,
        **options: Any,
    ) -> Job:
        warnings.warn(
            _LEGACY_DEPRECATION.format(name="from_audio_file"),
            DeprecationWarning,
            stacklevel=2,
        )
        return self.create(
            source_file=audio,
            include_ts=include_ts,
            model=model,
            return_result_in_response=return_result_in_response,
            webhook_url=webhook_url,
            **options,
        )

    def from_audio_file_price(
        self,
        *,
        include_ts: bool,
        model: str,
        audio: FileInput | None = None,
        duration_seconds: float | None = None,
        **options: Any,
    ) -> PriceResult:
        warnings.warn(
            _LEGACY_DEPRECATION.format(name="from_audio_file_price"),
            DeprecationWarning,
            stacklevel=2,
        )
        return self.create_price(
            include_ts=include_ts,
            model=model,
            source_file=audio,
            duration_seconds=duration_seconds,
            **options,
        )

    def from_video_file(
        self,
        *,
        video: FileInput,
        include_ts: bool,
        model: str,
        return_result_in_response: bool | None = None,
        webhook_url: str | None = None,
        **options: Any,
    ) -> Job:
        warnings.warn(
            _LEGACY_DEPRECATION.format(name="from_video_file"),
            DeprecationWarning,
            stacklevel=2,
        )
        return self.create(
            source_file=video,
            include_ts=include_ts,
            model=model,
            return_result_in_response=return_result_in_response,
            webhook_url=webhook_url,
            **options,
        )

    def from_video_file_price(
        self,
        *,
        include_ts: bool,
        model: str,
        video: FileInput | None = None,
        duration_seconds: float | None = None,
        **options: Any,
    ) -> PriceResult:
        warnings.warn(
            _LEGACY_DEPRECATION.format(name="from_video_file_price"),
            DeprecationWarning,
            stacklevel=2,
        )
        return self.create_price(
            include_ts=include_ts,
            model=model,
            source_file=video,
            duration_seconds=duration_seconds,
            **options,
        )


class AsyncTranscription:
    """Async transcription resource (v2)."""

    def __init__(self, client: AsyncHTTPClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        model: str,
        include_ts: bool | None = None,
        ts_level: str | None = None,
        diarize: bool | None = None,
        lang: str | None = None,
        include_metadata: bool | None = None,
        source_url: str | None = None,
        source_file: FileInput | None = None,
        return_result_in_response: bool | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
    ) -> AsyncJob:
        """Submit a transcription job. See ``Transcription.create`` for the parameters."""
        kwargs = _build_create(
            source_url=source_url, source_file=source_file, model=model, include_ts=include_ts,
            ts_level=ts_level, diarize=diarize, lang=lang, include_metadata=include_metadata,
            return_result_in_response=return_result_in_response,
            webhook_url=webhook_url, webhook_secret=webhook_secret,
        )
        resp = await self._client.post(self._client._resolve_endpoint("transcribe"), **kwargs)
        request_id = resp["data"]["request_id"]
        status_url = self._client._resolve_endpoint("request_status")
        return AsyncJob(request_id, self._client, status_url)

    async def create_price(
        self,
        *,
        model: str,
        include_ts: bool | None = None,
        ts_level: str | None = None,
        diarize: bool | None = None,
        lang: str | None = None,
        include_metadata: bool | None = None,
        platform: str | None = None,
        source_url: str | None = None,
        source_file: FileInput | None = None,
        duration_seconds: float | None = None,
    ) -> PriceResult:
        """Calculate the price of a transcription. See ``Transcription.create_price``."""
        kwargs = _build_price(
            source_url=source_url, source_file=source_file, model=model, include_ts=include_ts,
            ts_level=ts_level, diarize=diarize, lang=lang, include_metadata=include_metadata,
            platform=platform, duration_seconds=duration_seconds,
        )
        resp = await self._client.post(self._client._resolve_endpoint("transcribe_price"), **kwargs)
        return PriceResult.model_validate(resp.get("data", resp))

    async def get_transcript(
        self,
        result: JobResult,
        *,
        output_format: str | None = None,
        alt_format: str | None = None,
    ) -> Transcript:
        """Turn a finished job's result into a ``Transcript``.

        See ``Transcription.get_transcript``; the download sends no Authorization header.
        """
        inline, url = _pick_result_source(result, alt_format)
        if inline is None:
            async with httpx.AsyncClient(timeout=self._client._config.timeout, follow_redirects=True) as http:
                resp = await http.get(url)  # type: ignore[arg-type]
            self._client._raise_for_status(resp)
            inline = resp.text
        return parse_transcript(inline, output_format=_format_hint(output_format, alt_format))

    async def from_video_url(
        self,
        *,
        video_url: str,
        include_ts: bool,
        model: str,
        return_result_in_response: bool | None = None,
        webhook_url: str | None = None,
        **options: Any,
    ) -> AsyncJob:
        warnings.warn(
            _LEGACY_DEPRECATION.format(name="from_video_url"),
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.create(
            source_url=video_url,
            include_ts=include_ts,
            model=model,
            return_result_in_response=return_result_in_response,
            webhook_url=webhook_url,
            **options,
        )

    async def from_video_url_price(
        self,
        *,
        include_ts: bool,
        model: str,
        video_url: str | None = None,
        duration_seconds: float | None = None,
        **options: Any,
    ) -> PriceResult:
        warnings.warn(
            _LEGACY_DEPRECATION.format(name="from_video_url_price"),
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.create_price(
            include_ts=include_ts,
            model=model,
            source_url=video_url,
            duration_seconds=duration_seconds,
            **options,
        )

    async def from_audio_url(
        self,
        *,
        audio_url: str,
        include_ts: bool,
        model: str,
        return_result_in_response: bool | None = None,
        webhook_url: str | None = None,
        **options: Any,
    ) -> AsyncJob:
        warnings.warn(
            _LEGACY_DEPRECATION.format(name="from_audio_url"),
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.create(
            source_url=audio_url,
            include_ts=include_ts,
            model=model,
            return_result_in_response=return_result_in_response,
            webhook_url=webhook_url,
            **options,
        )

    async def from_audio_url_price(
        self,
        *,
        include_ts: bool,
        model: str,
        audio_url: str | None = None,
        duration_seconds: float | None = None,
        **options: Any,
    ) -> PriceResult:
        warnings.warn(
            _LEGACY_DEPRECATION.format(name="from_audio_url_price"),
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.create_price(
            include_ts=include_ts,
            model=model,
            source_url=audio_url,
            duration_seconds=duration_seconds,
            **options,
        )

    async def from_audio_file(
        self,
        *,
        audio: FileInput,
        include_ts: bool,
        model: str,
        return_result_in_response: bool | None = None,
        webhook_url: str | None = None,
        **options: Any,
    ) -> AsyncJob:
        warnings.warn(
            _LEGACY_DEPRECATION.format(name="from_audio_file"),
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.create(
            source_file=audio,
            include_ts=include_ts,
            model=model,
            return_result_in_response=return_result_in_response,
            webhook_url=webhook_url,
            **options,
        )

    async def from_audio_file_price(
        self,
        *,
        include_ts: bool,
        model: str,
        audio: FileInput | None = None,
        duration_seconds: float | None = None,
        **options: Any,
    ) -> PriceResult:
        warnings.warn(
            _LEGACY_DEPRECATION.format(name="from_audio_file_price"),
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.create_price(
            include_ts=include_ts,
            model=model,
            source_file=audio,
            duration_seconds=duration_seconds,
            **options,
        )

    async def from_video_file(
        self,
        *,
        video: FileInput,
        include_ts: bool,
        model: str,
        return_result_in_response: bool | None = None,
        webhook_url: str | None = None,
        **options: Any,
    ) -> AsyncJob:
        warnings.warn(
            _LEGACY_DEPRECATION.format(name="from_video_file"),
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.create(
            source_file=video,
            include_ts=include_ts,
            model=model,
            return_result_in_response=return_result_in_response,
            webhook_url=webhook_url,
            **options,
        )

    async def from_video_file_price(
        self,
        *,
        include_ts: bool,
        model: str,
        video: FileInput | None = None,
        duration_seconds: float | None = None,
        **options: Any,
    ) -> PriceResult:
        warnings.warn(
            _LEGACY_DEPRECATION.format(name="from_video_file_price"),
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.create_price(
            include_ts=include_ts,
            model=model,
            source_file=video,
            duration_seconds=duration_seconds,
            **options,
        )
