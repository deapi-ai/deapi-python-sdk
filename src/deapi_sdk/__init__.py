"""DeAPI Python SDK — Official client for the DeAPI distributed AI inference platform."""

from __future__ import annotations

import weakref
from typing import Any

import httpx

from deapi_sdk._client import AsyncHTTPClient, SyncHTTPClient
from deapi_sdk._config import ClientConfig
from deapi_sdk._exceptions import (
    AccountSuspendedError,
    ActiveKeyLimitReachedError,
    AuthenticationError,
    DeapiError,
    InsufficientBalanceError,
    InsufficientPermissionsError,
    IpNotAllowedError,
    JobTimeoutError,
    KeyLimitExceededError,
    NetworkError,
    NotFoundError,
    PayloadTooLargeError,
    PaymentRequiredError,
    PermissionDeniedError,
    RateLimitError,
    RealtimeError,
    ServerError,
    ValidationError,
)
from deapi_sdk._polling import AsyncJob, AsyncJobs, Job, Jobs
from deapi_sdk._realtime import AsyncRealtime, Realtime
from deapi_sdk._types import InferenceType, JobStatus
from deapi_sdk._version import __version__
from deapi_sdk.transcript import (
    Transcript,
    TranscriptSegment,
    TranscriptWord,
    parse_transcript,
    transcript_output_format,
)
from deapi_sdk.types.common import (
    Balance,
    JobResult,
    JobStatusEvent,
    JobSubmission,
    ModelInfo,
    ModelsResponse,
    PriceResult,
)
from deapi_sdk.types.v1.video import VideoDescription, VideoEvent, parse_video_description
from deapi_sdk.webhook import InvalidSignatureError, WebhookEvent, WebhookEventData, construct_event, verify_signature


def _attach_v1_resources_sync(client: DeapiClient) -> None:
    from deapi_sdk.resources.v1 import OCR, Audio, Embeddings, Images, Models, Prompts, Transcription, Video
    client.images = Images(client._http_client)
    client.models = Models(client._http_client)
    client.audio = Audio(client._http_client)
    client.transcription = Transcription(client._http_client)
    client.video = Video(client._http_client)
    client.embeddings = Embeddings(client._http_client)
    client.prompts = Prompts(client._http_client)
    client.ocr = OCR(client._http_client)
    client.jobs = Jobs(client._http_client)


def _attach_v1_resources_async(client: AsyncDeapiClient) -> None:
    from deapi_sdk.resources.v1 import (
        AsyncAudio,
        AsyncEmbeddings,
        AsyncImages,
        AsyncModels,
        AsyncOCR,
        AsyncPrompts,
        AsyncTranscription,
        AsyncVideo,
    )
    client.images = AsyncImages(client._http_client)
    client.models = AsyncModels(client._http_client)
    client.audio = AsyncAudio(client._http_client)
    client.transcription = AsyncTranscription(client._http_client)
    client.video = AsyncVideo(client._http_client)
    client.embeddings = AsyncEmbeddings(client._http_client)
    client.prompts = AsyncPrompts(client._http_client)
    client.ocr = AsyncOCR(client._http_client)
    client.jobs = AsyncJobs(client._http_client)


def _attach_v2_resources_sync(client: DeapiClient) -> None:
    from deapi_sdk.resources.v2 import (
        OCR,
        AccountResource,
        Audio,
        Embeddings,
        Images,
        KeysResource,
        Models,
        Prompts,
        Transcription,
        Video,
    )
    client.images = Images(client._http_client)
    client.models = Models(client._http_client)
    client.audio = Audio(client._http_client)
    client.transcription = Transcription(client._http_client)
    client.video = Video(client._http_client)
    client.embeddings = Embeddings(client._http_client)
    client.prompts = Prompts(client._http_client)
    client.ocr = OCR(client._http_client)
    client.jobs = Jobs(client._http_client)
    client.account = AccountResource(client._http_client)
    client.keys = KeysResource(client._http_client)


def _attach_v2_resources_async(client: AsyncDeapiClient) -> None:
    from deapi_sdk.resources.v2 import (
        AsyncAccountResource,
        AsyncAudio,
        AsyncEmbeddings,
        AsyncImages,
        AsyncKeysResource,
        AsyncModels,
        AsyncOCR,
        AsyncPrompts,
        AsyncTranscription,
        AsyncVideo,
    )
    client.images = AsyncImages(client._http_client)
    client.models = AsyncModels(client._http_client)
    client.audio = AsyncAudio(client._http_client)
    client.transcription = AsyncTranscription(client._http_client)
    client.video = AsyncVideo(client._http_client)
    client.embeddings = AsyncEmbeddings(client._http_client)
    client.prompts = AsyncPrompts(client._http_client)
    client.ocr = AsyncOCR(client._http_client)
    client.jobs = AsyncJobs(client._http_client)
    client.account = AsyncAccountResource(client._http_client)
    client.keys = AsyncKeysResource(client._http_client)


_RESOURCE_LOADERS: dict[str, dict[str, Any]] = {
    "v1": {
        "sync": _attach_v1_resources_sync,
        "async": _attach_v1_resources_async,
    },
    "v2": {
        "sync": _attach_v2_resources_sync,
        "async": _attach_v2_resources_async,
    },
}


class DeapiClient:
    """Synchronous DeAPI client.

    ``fill_model_defaults`` (default on; env ``DEAPI_FILL_MODEL_DEFAULTS=0`` disables): generation
    parameters you leave unset (width, height, steps, frames, fps, lang, ...) are taken from the
    model card's ``info.defaults`` (one catalogue request per client, cached), and ``seed`` defaults
    to -1 (random) where the API requires it. Pass ``False`` to send exactly what you set.

    Usage::

        client = DeapiClient(api_key="sk-...")
        job = client.images.generate(prompt="a cat in space", model="sdxl", width=1024, height=1024, seed=42)
        result = job.wait()
        print(result.result_url)

    Or as a context manager::

        with DeapiClient(api_key="sk-...") as client:
            ...
    """

    images: Any
    models: Any
    audio: Any
    transcription: Any
    video: Any
    embeddings: Any
    prompts: Any
    ocr: Any
    jobs: Any
    account: Any  # API v2 only
    keys: Any  # API v2 only

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        api_version: str | None = None,
        fill_model_defaults: bool | None = None,
        max_retry_after: float | None = None,
        http_client: httpx.Client | None = None,
        owns_http_client: bool | None = None,
    ) -> None:
        self._config = ClientConfig.from_env(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            api_version=api_version,
            fill_model_defaults=fill_model_defaults,
            max_retry_after=max_retry_after,
        )
        self._http_client = SyncHTTPClient(
            self._config, http_client, owns_http_client=owns_http_client
        )
        self._realtime_sessions: weakref.WeakSet[Realtime] = weakref.WeakSet()
        self._attach_resources()

    def _attach_resources(self) -> None:
        version = self._config.api_version
        loader = _RESOURCE_LOADERS.get(version)
        if loader is None:
            raise DeapiError(f"Unsupported API version: {version}")
        loader["sync"](self)

    def balance(self) -> Balance:
        """Get current account balance."""
        url = self._http_client._resolve_endpoint("balance")
        resp = self._http_client.get(url)
        return Balance.model_validate(resp.get("data", resp))

    def realtime(
        self,
        client_id: str | int | None = None,
        *,
        ws_url: str | None = None,
        app_key: str | None = None,
        connect_timeout: float = 30.0,
    ) -> Realtime:
        """Real-time job status updates over WebSocket (needs ``pip install "deapi-sdk[ws]"``).

        ``client_id`` is the Client ID shown in your deAPI dashboard; when omitted,
        ``DEAPI_CLIENT_ID`` is used. Servers acting for many users should always
        pass it explicitly. ``ws_url`` (``DEAPI_WS_URL``) and ``app_key``
        (``DEAPI_WS_APP_KEY``) default to the production server.

        The connection opens on ``with`` / ``connect()``. It is closed together
        with this client.
        """
        session = Realtime(
            self._http_client, client_id=client_id, ws_url=ws_url, app_key=app_key, connect_timeout=connect_timeout
        )
        self._realtime_sessions.add(session)
        return session

    def close(self) -> None:
        """Close open realtime sessions and the underlying HTTP client."""
        for session in list(self._realtime_sessions):
            session.close()
        self._http_client.close()

    def __enter__(self) -> DeapiClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AsyncDeapiClient:
    """Asynchronous DeAPI client.

    Takes the same ``fill_model_defaults`` option as :class:`DeapiClient`.

    Usage::

        async with AsyncDeapiClient(api_key="sk-...") as client:
            job = await client.images.generate(prompt="a cat in space", model="sdxl", width=1024, height=1024, seed=42)
            result = await job.wait()
            print(result.result_url)
    """

    images: Any
    models: Any
    audio: Any
    transcription: Any
    video: Any
    embeddings: Any
    prompts: Any
    ocr: Any
    jobs: Any
    account: Any  # API v2 only
    keys: Any  # API v2 only

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        api_version: str | None = None,
        fill_model_defaults: bool | None = None,
        max_retry_after: float | None = None,
        http_client: httpx.AsyncClient | None = None,
        owns_http_client: bool | None = None,
    ) -> None:
        self._config = ClientConfig.from_env(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            api_version=api_version,
            fill_model_defaults=fill_model_defaults,
            max_retry_after=max_retry_after,
        )
        self._http_client = AsyncHTTPClient(
            self._config, http_client, owns_http_client=owns_http_client
        )
        self._realtime_sessions: weakref.WeakSet[AsyncRealtime] = weakref.WeakSet()
        self._attach_resources()

    def _attach_resources(self) -> None:
        version = self._config.api_version
        loader = _RESOURCE_LOADERS.get(version)
        if loader is None:
            raise DeapiError(f"Unsupported API version: {version}")
        loader["async"](self)

    async def balance(self) -> Balance:
        """Get current account balance."""
        url = self._http_client._resolve_endpoint("balance")
        resp = await self._http_client.get(url)
        return Balance.model_validate(resp.get("data", resp))

    def realtime(
        self,
        client_id: str | int | None = None,
        *,
        ws_url: str | None = None,
        app_key: str | None = None,
        connect_timeout: float = 30.0,
    ) -> AsyncRealtime:
        """Real-time job status updates over WebSocket (needs ``pip install "deapi-sdk[ws]"``).

        ``client_id`` is the Client ID shown in your deAPI dashboard; when omitted,
        ``DEAPI_CLIENT_ID`` is used. Servers acting for many users should always
        pass it explicitly. ``ws_url`` (``DEAPI_WS_URL``) and ``app_key``
        (``DEAPI_WS_APP_KEY``) default to the production server.

        The connection opens on ``async with`` /
        ``await connect()``. It is closed together
        with this client.
        """
        session = AsyncRealtime(
            self._http_client, client_id=client_id, ws_url=ws_url, app_key=app_key, connect_timeout=connect_timeout
        )
        self._realtime_sessions.add(session)
        return session

    async def close(self) -> None:
        """Close open realtime sessions and the underlying HTTP client."""
        for session in list(self._realtime_sessions):
            await session.close()
        await self._http_client.close()

    async def __aenter__(self) -> AsyncDeapiClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


__all__ = [
    # Clients
    "DeapiClient",
    "AsyncDeapiClient",
    # Jobs
    "Job",
    "AsyncJob",
    "AsyncJobs",
    "Jobs",
    # Realtime (WebSocket)
    "AsyncRealtime",
    "Realtime",
    "JobStatusEvent",
    # Types
    "Balance",
    "InferenceType",
    "JobResult",
    "JobStatus",
    "JobSubmission",
    "ModelInfo",
    "ModelsResponse",
    "PriceResult",
    # Results
    "Transcript",
    "TranscriptSegment",
    "TranscriptWord",
    "VideoDescription",
    "VideoEvent",
    "parse_transcript",
    "parse_video_description",
    "transcript_output_format",
    # Exceptions
    "AccountSuspendedError",
    "ActiveKeyLimitReachedError",
    "AuthenticationError",
    "NetworkError",
    "DeapiError",
    "InsufficientBalanceError",
    "InsufficientPermissionsError",
    "IpNotAllowedError",
    "JobTimeoutError",
    "KeyLimitExceededError",
    "PayloadTooLargeError",
    "PaymentRequiredError",
    "PermissionDeniedError",
    "NotFoundError",
    "RateLimitError",
    "RealtimeError",
    "ServerError",
    "ValidationError",
    # Webhook
    "InvalidSignatureError",
    "WebhookEvent",
    "WebhookEventData",
    "construct_event",
    "verify_signature",
    # Version
    "__version__",
]
