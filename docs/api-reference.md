# DeAPI Python SDK — API Reference

Complete reference for all classes, methods, and types in the `deapi` package.

## Table of Contents

- [Client Classes](#client-classes)
- [Resource: images](#resource-images)
- [Resource: audio](#resource-audio)
- [Resource: video](#resource-video)
- [Resource: transcription](#resource-transcription)
- [Resource: embeddings](#resource-embeddings)
- [Resource: prompts](#resource-prompts)
- [Resource: models](#resource-models)
- [Job Polling](#job-polling)
- [Types & Models](#types--models)
- [Exceptions](#exceptions)
- [Webhook Utilities](#webhook-utilities)

---

## Client Classes

### `DeapiClient`

Synchronous client for the DeAPI platform.

```python
from deapi import DeapiClient

client = DeapiClient(
    *,
    api_key: str | None = None,        # Falls back to DEAPI_API_KEY env var
    base_url: str | None = None,        # Falls back to DEAPI_BASE_URL (default: "https://api.deapi.ai")
    timeout: float | None = None,       # Request timeout in seconds (default: 30.0)
    max_retries: int | None = None,     # Retry attempts for 429/5xx/network errors (default: 3)
    api_version: str | None = None,     # Falls back to DEAPI_API_VERSION (default: "v1")
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `client.balance()` | `Balance` | Get current account balance |
| `client.close()` | `None` | Close the underlying HTTP connection |

**Resource namespaces:** `client.images`, `client.audio`, `client.video`, `client.transcription`, `client.embeddings`, `client.models`, `client.prompts`

**Context manager:**

```python
with DeapiClient(api_key="sk-...") as client:
    ...  # client.close() called automatically
```

### `AsyncDeapiClient`

Asynchronous client — identical API surface, all methods are `async`.

```python
from deapi import AsyncDeapiClient

async with AsyncDeapiClient(api_key="sk-...") as client:
    balance = await client.balance()
```

---

## Resource: images

Access via `client.images`.

### `generate()`

Generate an image from a text prompt (text-to-image).

```python
job = client.images.generate(
    *,
    prompt: str,                                           # Required
    model: str,                                            # Required
    width: int,                                            # Required
    height: int,                                           # Required
    seed: int,                                             # Required
    negative_prompt: str | None = None,
    loras: list[LoraWeight | dict] | None = None,          # [{"name": "...", "weight": 0.8}]
    guidance: float | None = None,
    steps: int | None = None,
    webhook_url: str | None = None,
) -> Job
```

### `generate_price()`

Calculate the price for a text-to-image generation. Same parameters as `generate()` except `webhook_url`.

```python
price = client.images.generate_price(...) -> PriceResult
```

### `transform()`

Transform an existing image using a text prompt (image-to-image). Accepts file uploads.

```python
job = client.images.transform(
    *,
    prompt: str,                                           # Required
    model: str,                                            # Required
    steps: int,                                            # Required
    seed: int,                                             # Required
    image: FileInput | None = None,                        # Single image (mutually exclusive with images)
    images: list[FileInput] | None = None,                 # Multiple images
    negative_prompt: str | None = None,
    loras: list[LoraWeight | dict] | None = None,
    width: int | None = None,
    height: int | None = None,
    guidance: float | None = None,
    webhook_url: str | None = None,
) -> Job
```

`FileInput` accepts: `str` (path), `Path`, `bytes`, or `BinaryIO` (file-like object).

### `transform_price()`

```python
price = client.images.transform_price(
    *, prompt: str, model: str, steps: int, seed: int,
    loras: list | None = None, guidance: float | None = None,
) -> PriceResult
```

### `upscale()`

Upscale an image to higher resolution.

```python
job = client.images.upscale(
    *,
    image: FileInput,                                      # Required
    model: str,                                            # Required
    webhook_url: str | None = None,
) -> Job
```

### `upscale_price()`

```python
price = client.images.upscale_price(
    *, model: str,
    image: FileInput | None = None,                        # Provide image OR width+height
    width: int | None = None,
    height: int | None = None,
) -> PriceResult
```

### `remove_background()`

Remove the background from an image.

```python
job = client.images.remove_background(
    *,
    image: FileInput,                                      # Required
    model: str,                                            # Required
    webhook_url: str | None = None,
) -> Job
```

### `remove_background_price()`

```python
price = client.images.remove_background_price(
    *, model: str,
    image: FileInput | None = None,                        # Provide image OR width+height
    width: int | None = None,
    height: int | None = None,
) -> PriceResult
```

---

## Resource: audio

Access via `client.audio`.

### `synthesize()`

Convert text to speech (TTS). Supports voice cloning via `ref_audio`.

```python
job = client.audio.synthesize(
    *,
    text: str,                                             # Required
    model: str,                                            # Required
    lang: str,                                             # Required — language code (e.g. "en-us")
    format: str,                                           # Required — "wav", "flac", or "mp3"
    speed: float,                                          # Required — speech speed (min 0.5)
    sample_rate: float,                                    # Required — Hz (model-specific)
    mode: str | None = None,                               # "custom_voice", "voice_clone", or "voice_design"
    voice: str | None = None,                              # Required when mode is "custom_voice"
    ref_audio: FileInput | None = None,                    # Required for "voice_clone" mode
    ref_text: str | None = None,                           # Optional reference text for voice cloning
    instruct: str | None = None,                           # Required for "voice_design" mode
    webhook_url: str | None = None,
) -> Job
```

### `synthesize_price()`

```python
price = client.audio.synthesize_price(
    *, model: str, lang: str, format: str, speed: float, sample_rate: float,
    text: str | None = None,                               # Provide text OR count_text
    count_text: int | None = None,                         # Character count (skip sending full text)
    mode: str | None = None,
    voice: str | None = None,
    instruct: str | None = None,
) -> PriceResult
```

### `compose()`

Generate music from a text description.

```python
job = client.audio.compose(
    *,
    caption: str,                                          # Required — music description
    model: str,                                            # Required
    duration: float,                                       # Required — seconds (10–600)
    inference_steps: int,                                  # Required
    guidance_scale: float,                                 # Required
    seed: int,                                             # Required
    format: str,                                           # Required — "wav", "flac", or "mp3"
    lyrics: str | None = None,
    bpm: int | None = None,                                # 30–300
    keyscale: str | None = None,
    timesignature: int | None = None,                      # 2, 3, 4, or 6
    vocal_language: str | None = None,
    webhook_url: str | None = None,
) -> Job
```

### `compose_price()`

```python
price = client.audio.compose_price(
    *, model: str, duration: float, inference_steps: int,
) -> PriceResult
```

---

## Resource: video

Access via `client.video`.

### `generate()`

Generate a video from a text prompt (text-to-video).

```python
job = client.video.generate(
    *,
    prompt: str,                                           # Required
    model: str,                                            # Required
    width: int,                                            # Required
    height: int,                                           # Required
    steps: int,                                            # Required
    seed: int,                                             # Required
    frames: int,                                           # Required
    fps: int,                                              # Required
    negative_prompt: str | None = None,
    guidance: float | None = None,
    webhook_url: str | None = None,
) -> Job
```

### `generate_price()`

Same parameters as `generate()` except `webhook_url`. Returns `PriceResult`.

### `animate()`

Animate a still image into a video (image-to-video).

```python
job = client.video.animate(
    *,
    prompt: str,                                           # Required
    model: str,                                            # Required
    first_frame_image: FileInput,                          # Required
    seed: int,                                             # Required
    width: int,                                            # Required
    height: int,                                           # Required
    frames: int,                                           # Required
    fps: int,                                              # Required
    negative_prompt: str | None = None,
    last_frame_image: FileInput | None = None,
    guidance: float | None = None,
    steps: int | None = None,
    webhook_url: str | None = None,
) -> Job
```

### `animate_price()`

```python
price = client.video.animate_price(
    *, model: str, width: int, height: int, frames: int, fps: int,
    steps: int | None = None, guidance: float | None = None,
) -> PriceResult
```

### `generate_from_audio()`

Generate a video synced to an audio track (audio-to-video).

```python
job = client.video.generate_from_audio(
    *,
    prompt: str,                                           # Required
    audio: FileInput,                                      # Required — MP3, WAV, OGG, or FLAC
    model: str,                                            # Required
    width: int,                                            # Required
    height: int,                                           # Required
    seed: int,                                             # Required
    frames: int,                                           # Required
    fps: int,                                              # Required
    negative_prompt: str | None = None,
    first_frame_image: FileInput | None = None,
    last_frame_image: FileInput | None = None,
    guidance: float | None = None,
    steps: int | None = None,
    webhook_url: str | None = None,
) -> Job
```

### `generate_from_audio_price()`

```python
price = client.video.generate_from_audio_price(
    *, model: str, width: int, height: int, frames: int, fps: int,
    guidance: float | None = None, steps: int | None = None,
) -> PriceResult
```

### `replace()`

Replace a character in a video using a reference image.

```python
job = client.video.replace(
    *,
    video: FileInput,                                      # Required — input video
    ref_image: FileInput,                                  # Required — reference character image
    model: str,                                            # Required
    prompt: str | None = None,
    width: int | None = None,                              # Must be paired with height
    height: int | None = None,                             # Must be paired with width
    steps: int | None = None,                              # Default: 4
    seed: int | None = None,                               # Default: -1
    webhook_url: str | None = None,
) -> Job
```

### `replace_price()`

```python
price = client.video.replace_price(
    *, model: str,
    video: FileInput | None = None,                        # Provide video OR duration
    duration: float | None = None,                         # Duration in seconds (alternative to video)
    width: int | None = None,
    height: int | None = None,
) -> PriceResult
```

> Provide either `video` (file) or `duration` (seconds). Raises `ValueError` if neither is given.

### `upscale()`

Upscale a video to higher resolution.

```python
job = client.video.upscale(*, video: FileInput, model: str, webhook_url: str | None = None) -> Job
```

### `upscale_price()`

```python
price = client.video.upscale_price(
    *, model: str, video: FileInput | None = None, width: int | None = None, height: int | None = None,
) -> PriceResult
```

### `remove_background()`

Remove background from a video.

```python
job = client.video.remove_background(*, video: FileInput, model: str, webhook_url: str | None = None) -> Job
```

### `remove_background_price()`

```python
price = client.video.remove_background_price(
    *, model: str, video: FileInput | None = None, width: int | None = None, height: int | None = None,
) -> PriceResult
```

---

## Resource: transcription

Access via `client.transcription`.

### `create()` (Recommended)

Unified transcription endpoint — auto-detects source type.

```python
job = client.transcription.create(
    *,
    include_ts: bool,                                      # Required — include timestamps
    model: str,                                            # Required
    source_url: str | None = None,                         # URL (YouTube, Twitch, Twitter, etc.)
    source_file: FileInput | None = None,                  # Audio or video file upload
    return_result_in_response: bool | None = None,         # Inline result in status response
    webhook_url: str | None = None,
) -> Job
```

> Provide either `source_url` or `source_file`, not both.

Auto-detection:
- Twitter Spaces URL → Audio transcription
- Other URL → Video transcription
- Audio MIME file → Audio file transcription
- Video MIME file → Video file transcription

### `create_price()`

```python
price = client.transcription.create_price(
    *, include_ts: bool, model: str,
    source_url: str | None = None,
    source_file: FileInput | None = None,
    duration_seconds: float | None = None,                 # Skip source, use known duration
) -> PriceResult
```

### Legacy Methods

These target specific endpoints directly:

| Method | Description |
|--------|-------------|
| `from_video_url(*, video_url, include_ts, model, ...)` | Transcribe from video URL |
| `from_video_url_price(...)` | Price for video URL transcription |
| `from_audio_url(*, audio_url, include_ts, model, ...)` | Transcribe from audio URL (Twitter Spaces) |
| `from_audio_url_price(...)` | Price for audio URL transcription |
| `from_audio_file(*, audio, include_ts, model, ...)` | Transcribe from audio file upload |
| `from_audio_file_price(...)` | Price for audio file transcription |
| `from_video_file(*, video, include_ts, model, ...)` | Transcribe from video file upload |
| `from_video_file_price(...)` | Price for video file transcription |

All legacy methods return `Job` and their `_price` variants return `PriceResult`.

---

## Resource: embeddings

Access via `client.embeddings`.

### `create()`

Generate text embeddings (single or batch).

```python
job = client.embeddings.create(
    *,
    input: str | list[str],                                # Required — single text or batch
    model: str,                                            # Required
    return_result_in_response: bool | None = None,
    webhook_url: str | None = None,
) -> Job
```

### `create_price()`

```python
price = client.embeddings.create_price(
    *, input: str | list[str], model: str,
    return_result_in_response: bool | None = None,
) -> PriceResult
```

---

## Resource: prompts

Access via `client.prompts`.

> Prompt methods return results **immediately** — no job polling needed.

### `enhance_image()`

AI-enhance a text-to-image prompt.

```python
result = client.prompts.enhance_image(
    *, prompt: str, negative_prompt: str | None = None,
) -> EnhancePromptResult
```

### `enhance_video()`

AI-enhance a video generation prompt, optionally with a reference image.

```python
result = client.prompts.enhance_video(
    *, prompt: str, negative_prompt: str | None = None, image: FileInput | None = None,
) -> EnhancePromptResult
```

### `enhance_speech()`

AI-enhance a text-to-speech prompt.

```python
result = client.prompts.enhance_speech(
    *, prompt: str, lang_code: str | None = None,
) -> EnhanceSpeechPromptResult
```

### `enhance_image2image()`

AI-enhance an image-to-image prompt with a reference image.

```python
result = client.prompts.enhance_image2image(
    *, prompt: str, image: FileInput, negative_prompt: str | None = None,
) -> EnhancePromptResult
```

### `samples()`

Get a sample prompt for inspiration.

```python
result = client.prompts.samples(
    *, type: str,                                          # "image" or "speech"
    topic: str | None = None,
    lang_code: str | None = None,
) -> SamplePromptResult
```

### Price Methods

Every prompt method has a `_price` variant: `enhance_image_price()`, `enhance_video_price()`, `enhance_speech_price()`, `enhance_image2image_price()`, `samples_price()`. All return `PriceResult`.

---

## Resource: models

Access via `client.models`.

### `list()`

List available AI models with their capabilities.

```python
response = client.models.list(
    *,
    per_page: int = 15,                                    # Max 50
    page: int = 1,
    inference_types: list[str] | None = None,              # Filter, e.g. ["txt2img", "img2img"]
) -> ModelsResponse
```

---

## Job Polling

All generation/submission methods return a `Job` (sync) or `AsyncJob` (async).

### `Job`

```python
class Job:
    request_id: str                       # The unique job ID

    def status() -> JobResult             # Poll once (makes an HTTP request)
    def wait(
        *,
        poll_interval: float = 1.0,       # Initial seconds between polls
        max_wait: float = 300.0,          # Total timeout in seconds
        backoff_factor: float = 1.5,      # Multiplier for backoff
        max_interval: float = 10.0,       # Cap on poll interval
    ) -> JobResult                        # Blocks until done/error
    def is_done() -> bool                 # Convenience (makes an HTTP request)
    def is_error() -> bool                # Convenience (makes an HTTP request)
```

> **Note:** `is_done()` and `is_error()` each make an HTTP request. For efficient polling, use `status()` once and check the result.

### `AsyncJob`

Same interface, all methods are `async`.

---

## Types & Models

### `JobResult`

Returned by `job.status()` and `job.wait()`.

```python
class JobResult(BaseModel):
    status: JobStatus                     # "pending", "processing", "done", "error"
    preview: str | None                   # Preview URL (during processing)
    result_url: str | None                # Final output URL (when done)
    results_alt_formats: dict[str, str] | None  # {"jpg": "...", "webp": "..."} (image jobs)
    result: str | None                    # Inline text result (when return_result_in_response=True)
    progress: float                       # 0.0–100.0
```

### `PriceResult`

```python
class PriceResult(BaseModel):
    price: float                          # Cost in credits
```

### `Balance`

```python
class Balance(BaseModel):
    balance: float                        # Current account balance
```

### `ModelsResponse`

```python
class ModelsResponse(BaseModel):
    data: list[ModelInfo]
    links: PaginationLinks
    meta: PaginationMeta

class ModelInfo(BaseModel):
    name: str
    inference_types: list[str]
    specs: dict | None

class PaginationMeta(BaseModel):
    current_page: int
    last_page: int
    per_page: int
    total: int
```

### `EnhancePromptResult`

```python
class EnhancePromptResult(BaseModel):
    prompt: str
    negative_prompt: str | None
```

### `EnhanceSpeechPromptResult`

```python
class EnhanceSpeechPromptResult(BaseModel):
    prompt: str
```

### `SamplePromptResult`

```python
class SamplePromptResult(BaseModel):
    type: str
    prompt: str
```

### Enums

```python
class JobStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    ERROR      = "error"

class InferenceType(str, Enum):
    TXT2IMG          = "txt2img"
    IMG2IMG          = "img2img"
    TXT2AUDIO        = "txt2audio"
    TXT2MUSIC        = "txt2music"
    # ... and more (see deapi._types for full list)
```

---

## Exceptions

All exceptions inherit from `DeapiError`.

```
DeapiError                          # Base — attrs: message, status_code, body
├── AuthenticationError             # 401 — invalid/missing API key
├── AccountSuspendedError           # 403 — account suspended
├── NotFoundError                   # 404 — resource not found
├── ValidationError                 # 422 — invalid parameters
│   └── InsufficientBalanceError    # 422 with balance error
├── RateLimitError                  # 429 — rate limited
├── ServerError                     # 5xx — server error
├── JobTimeoutError                 # Polling exceeded max_wait
└── NetworkError                    # Connection/timeout failure
```

### `DeapiError`

```python
class DeapiError(Exception):
    message: str
    status_code: int | None
    body: dict | None
```

### `ValidationError`

```python
class ValidationError(DeapiError):
    errors: dict[str, list[str]]    # {"field": ["error message", ...]}
```

### `RateLimitError`

```python
class RateLimitError(DeapiError):
    retry_after: float              # Seconds to wait (from Retry-After header)
    limit_type: str                 # "minute" or "daily"
```

### Auto-Retry Behavior

The client automatically retries on:

| Error | Behavior |
|-------|----------|
| `RateLimitError` (429) | Waits `Retry-After` seconds, retries up to `max_retries` |
| `ServerError` (5xx) | Exponential backoff (1s, 2s, 4s, 8s), up to `max_retries` |
| `NetworkError` | Exponential backoff (1s, 2s, 4s, 8s), up to `max_retries` |

---

## Webhook Utilities

```python
from deapi.webhook import verify_signature, construct_event, InvalidSignatureError
```

### `verify_signature()`

Verify a webhook request's HMAC-SHA256 signature. Raises `InvalidSignatureError` on failure.

```python
verify_signature(
    *,
    payload: bytes | str,           # Raw request body
    signature: str,                 # X-DeAPI-Signature header
    timestamp: str,                 # X-DeAPI-Timestamp header
    secret: str,                    # Your webhook signing secret
    tolerance: int = 300,           # Max age in seconds (0 = disable)
) -> None
```

### `construct_event()`

Verify signature and parse the webhook payload into a typed object.

```python
event = construct_event(
    *,
    payload: bytes | str,
    signature: str,
    timestamp: str,
    secret: str,
    tolerance: int = 300,
) -> WebhookEvent
```

### `WebhookEvent`

```python
class WebhookEvent(BaseModel):
    event: str                      # "job.completed", "job.processing", "job.failed"
    delivery_id: str
    timestamp: str
    data: WebhookEventData

    @property
    def type(self) -> str           # Alias for event

class WebhookEventData(BaseModel):
    job_request_id: str
    status: str
    previous_status: str | None
    job_type: str | None
    completed_at: str | None
    result_url: str | None
    processing_time_ms: int | None
```

### `InvalidSignatureError`

```python
class InvalidSignatureError(DeapiError):
    ...  # Raised by verify_signature / construct_event
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEAPI_API_KEY` | *(required)* | API key for authentication |
| `DEAPI_BASE_URL` | `https://api.deapi.ai` | API base URL |
| `DEAPI_API_VERSION` | `v1` | API version |
