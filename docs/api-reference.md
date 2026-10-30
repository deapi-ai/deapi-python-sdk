# DeAPI Python SDK — API Reference

Reference for the classes, methods and types in the `deapi_sdk` package. It describes deAPI API **v2**, the only version supported since 0.4.0 (see
[Upgrading to 0.4.0](../README.md#upgrading-to-040) for the breaking changes). The HTTP API itself is documented at <https://docs.deapi.ai>.

Install: `pip install git+https://github.com/deapi-ai/deapi-python-sdk.git` (not on PyPI yet;
`pip install deapi` is an unrelated package).

## Table of Contents

- [Client Classes](#client-classes)
- [Conventions](#conventions)
- [Resource: images](#resource-images)
- [Resource: audio](#resource-audio)
- [Resource: video](#resource-video)
- [Resource: transcription](#resource-transcription)
- [Resource: embeddings](#resource-embeddings)
- [Resource: ocr](#resource-ocr)
- [Resource: prompts](#resource-prompts)
- [Resource: models](#resource-models)
- [Resource: jobs](#resource-jobs)
- [Resource: account](#resource-account)
- [Resource: keys](#resource-keys)
- [Job Polling](#job-polling)
- [Realtime (WebSocket)](#realtime-websocket)
- [Types & Models](#types--models)
- [Exceptions](#exceptions)
- [Webhook Utilities](#webhook-utilities)
- [Environment Variables](#environment-variables)

---

## Client Classes

### `DeapiClient`

```python
from deapi_sdk import DeapiClient

client = DeapiClient(
    *,
    api_key: str | None = None,            # Falls back to DEAPI_API_KEY. Format: sk-dp-<id>-<secret>
    base_url: str | None = None,           # Falls back to DEAPI_BASE_URL (default "https://api.deapi.ai")
    timeout: float | None = None,          # Request timeout in seconds (default 30.0)
    max_retries: int | None = None,        # Retry attempts (default 3), see "Retry policy"
    api_version: str | None = None,        # Falls back to DEAPI_API_VERSION (only "v2" is accepted, default)
    fill_model_defaults: bool | None = None,  # Falls back to DEAPI_FILL_MODEL_DEFAULTS (default on)
)
```

API keys look like `sk-dp-<id>-<secret>`; older `<id>|<secret>` keys keep working. Keys have a
role: ADMIN keys can use every endpoint, API keys get 403 `insufficient_permissions` on balance,
account and key-management endpoints.

| Method | Returns | Description |
|--------|---------|-------------|
| `client.balance()` | `Balance` | Account balance (ADMIN key) |
| `client.close()` | `None` | Close the HTTP connection |

**Resources:** `images`, `audio`, `video`, `transcription`, `embeddings`, `ocr`, `prompts`, `models`,
`jobs`, `account` and `keys`.

```python
with DeapiClient() as client:
    ...  # client.close() on exit
```

### `AsyncDeapiClient`

Same API surface, every method is `async`.

```python
from deapi_sdk import AsyncDeapiClient

async with AsyncDeapiClient() as client:
    balance = await client.balance()
```

### Model-card defaults

Parameters that depend on the model (`width`, `height`, `steps`, `frames`, `fps`, `seed`, `lang`,
`speed`, `format`, `sample_rate`, `inference_steps`, `guidance_scale`, ...) are optional in the
method signatures. When one is unset, the SDK fills it from the model's `info.defaults` in
`GET /models` (one catalogue request per client, cached) and `seed` defaults to `-1` (random).
Explicit values always win. Disable with `DeapiClient(fill_model_defaults=False)` or
`DEAPI_FILL_MODEL_DEFAULTS=0`; the API then applies its own defaults.

---

## Conventions

- `FileInput` is `str` (path), `Path`, `bytes` or a binary file-like object.
- Every job method returns a `Job` and has a `*_price` counterpart returning `PriceResult`. Price
  methods take the same model parameters (not `webhook_*`). The `enhance_prompt` fee is billed on
  the job and is **not** included in a price quote.
- **Webhook parameters** on every job method: `webhook_url: str | None` (must be `https`) and
  `webhook_secret: str | None` (32-255 characters, requires `webhook_url`). The secret is a per-job
  HMAC key used to sign that job's deliveries; verify with [`construct_event`](#webhook-utilities).
  Invalid combinations raise `ValueError` before any request is sent.
- Methods are keyword-only.

---

## Resource: images

### `generate()`

Text-to-image.

```python
job = client.images.generate(
    *,
    prompt: str,
    model: str,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
    steps: int | None = None,
    guidance: float | None = None,
    negative_prompt: str | None = None,
    loras: list[LoraWeight | dict] | None = None,   # [{"name": "...", "weight": 0.8}]
    quality: str | None = None,                     # model tier, e.g. "low" | "medium" | "high"
    enhance_prompt: bool | None = None,             # boost the prompt inside the job
    webhook_url: str | None = None,
    webhook_secret: str | None = None,
) -> Job
```

`generate_price()` takes the same parameters except `enhance_prompt` and `webhook_*`.

### `transform()`

Image edit (image-to-image).

```python
job = client.images.transform(
    *,
    prompt: str,
    model: str,
    image: FileInput | None = None,          # one source ...
    images: list[FileInput] | None = None,   # ... or several (sent as images[]); not both
    steps: int | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
    loras: list[LoraWeight | dict] | None = None,
    width: int | None = None,
    height: int | None = None,
    guidance: float | None = None,
    quality: str | None = None,
    enhance_prompt: bool | None = None,
    webhook_url: str | None = None,
    webhook_secret: str | None = None,
) -> Job
```

The per-model maximum number of sources is `max_input_images` on the model card.
`transform_price()` takes the same parameters (pass the same `image` / `images`: some models price
by input size).

### `upscale()` / `remove_background()`

```python
client.images.upscale(*, image: FileInput, model: str, webhook_url=None, webhook_secret=None) -> Job
client.images.remove_background(*, image: FileInput, model: str, webhook_url=None, webhook_secret=None) -> Job
```

`upscale_price()` / `remove_background_price()`: `model`, plus `image` **or** `width` + `height`.

---

## Resource: audio

### `synthesize()`

Text-to-speech, with voice cloning via `ref_audio`.

```python
job = client.audio.synthesize(
    *,
    text: str,
    model: str,
    lang: str | None = None,            # e.g. "en-us"
    format: str | None = None,          # "wav" | "flac" | "mp3"
    speed: float | None = None,
    sample_rate: float | None = None,
    mode: str | None = None,            # "custom_voice" | "voice_clone" | "voice_design"
    voice: str | None = None,           # for "custom_voice"
    ref_audio: FileInput | None = None, # for "voice_clone"
    ref_text: str | None = None,
    instruct: str | None = None,        # for "voice_design"
    webhook_url: str | None = None,
    webhook_secret: str | None = None,
) -> Job
```

`synthesize_price()`: same, but pass `text` **or** `count_text` (character count), no `ref_*`.

### `compose()`

Text-to-music, optionally conditioned on a style reference clip.

```python
job = client.audio.compose(
    *,
    caption: str,
    model: str,
    duration: float,                        # seconds
    inference_steps: int | None = None,
    guidance_scale: float | None = None,
    seed: int | None = None,
    format: str | None = None,              # "wav" | "flac" | "mp3"
    lyrics: str | None = None,              # use "[Instrumental]" for instrumentals
    bpm: int | None = None,                 # 30-300
    keyscale: str | None = None,
    timesignature: int | None = None,       # 2, 3, 4 or 6
    vocal_language: str | None = None,
    reference_audio: FileInput | None = None,   # style reference; switches the request to multipart
    webhook_url: str | None = None,
    webhook_secret: str | None = None,
) -> Job
```

`compose_price()`: `model`, `duration`, and optionally `inference_steps`, `guidance_scale`,
`caption`, `lyrics`, `bpm`, `keyscale`, `timesignature`, `vocal_language`.

---

## Resource: video

### `generate()`

Text-to-video.

```python
job = client.video.generate(
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
    enhance_prompt: bool | None = None,     # needs a prompt-booster guide for the model
    webhook_url: str | None = None,
    webhook_secret: str | None = None,
) -> Job
```

### `animate()`

Image-to-video. Two input modes, pick one:

- **Keyframes**: `first_frame_image` and, on models that support it, `last_frame_image`.
- **Reference mode**: `ref_images` and/or `ref_videos`, optionally with `ref_audios`. Mutually
  exclusive with first/last frame; audio alone is not enough.

```python
job = client.video.animate(
    *,
    prompt: str,
    model: str,
    first_frame_image: FileInput | None = None,
    last_frame_image: FileInput | None = None,
    ref_images: list[FileInput] | None = None,
    ref_videos: list[FileInput] | None = None,
    ref_audios: list[FileInput] | None = None,
    width: int | None = None, height: int | None = None,
    frames: int | None = None, fps: int | None = None,
    steps: int | None = None, seed: int | None = None,
    negative_prompt: str | None = None, guidance: float | None = None,
    enhance_prompt: bool | None = None,
    webhook_url: str | None = None, webhook_secret: str | None = None,
) -> Job
```

List order binds files to prompt tags: `ref_images[0]` is `<Picture 1>`, `ref_videos[0]` is
`<Video 1>`, `ref_audios[0]` is `<Audio 1>` (`<Audio 2>` if the reference video has its own audio
track). Numbering is independent per type. Limits come from the model card (`max_ref_images`,
`max_ref_videos`, `max_ref_audios`, `max_ref_files`); the API never accepts more than 9 images, 1
video or 3 audios. Raises `ValueError` for mixed modes, audio-only references or no image source.

`animate_price()`: `model`, `width`, `height`, `frames`, `fps`, `steps`, `guidance`, plus the
*counts* `ref_images_count` (0-9), `ref_videos_count` (0-1), `ref_audios_count` (0-3). More than 2
reference images apply a x2 multiplier, any reference video x3 (highest wins).

### `generate_from_audio()`

Audio-to-video: `prompt`, `audio` (MP3/WAV/OGG/FLAC), `model`, optional `first_frame_image`,
`last_frame_image`, `width`, `height`, `frames`, `fps`, `steps`, `seed`, `negative_prompt`,
`guidance`, webhook params. `generate_from_audio_price()` takes the sizing parameters.

### `describe()` / `describe_price()` / `get_description()`

Timestamped index of what happens on screen (reads the picture, not the audio).

```python
job = client.video.describe(
    *,
    model: str,                                  # e.g. "Marlin_2B"
    video_url: str | None = None,                # exactly one of video_url / video_file
    video_file: FileInput | None = None,
    preset: str | None = None,                   # "fast" | "balanced" | "detail" (models with supports_presets)
    include_metadata: bool | None = None,        # URL sources only
    return_result_in_response: bool | None = None,
    webhook_url: str | None = None, webhook_secret: str | None = None,
) -> Job

price = client.video.describe_price(
    *, model, video_url=None, video_file=None, duration_seconds=None,
    platform=None,      # youtube | twitter | twitch | kick | tiktok; only with duration_seconds
    preset=None, include_metadata=None,
) -> PriceResult

description = client.video.get_description(result: JobResult | str) -> VideoDescription
```

`VideoDescription` has `events`, `scenes` and `meta`. `get_description` uses the inline result when
present, else downloads `result_url` without the API key.

### `upscale()` / `remove_background()`

```python
client.video.upscale(*, video: FileInput, model: str, webhook_url=None, webhook_secret=None) -> Job
client.video.remove_background(*, video: FileInput, model: str, webhook_url=None, webhook_secret=None) -> Job
```

Price variants take `model` plus `video` **or** `width` + `height` + `duration` (seconds). `upscale()` takes `scale` (2 or 4) on models whose card has a `scale` range, e.g. FlashVSR_Tiny.

### `replace()`

Kept for compatibility (`video`, `ref_image`, `model`, optional `prompt`, `width`, `height`,
`steps`, `seed`, webhook params; `replace_price()` takes `video` or `duration`). **No model
currently serves this route.** Use `animate()` with the character as `ref_images` and the source
motion as `ref_videos` instead.

---

## Resource: transcription

### `create()`

Unified endpoint for URLs and file uploads.

```python
job = client.transcription.create(
    *,
    model: str,
    source_url: str | None = None,              # exactly one of source_url / source_file
    source_file: FileInput | None = None,
    include_ts: bool | None = None,             # implied by ts_level / diarize
    ts_level: str | None = None,                # "segment" | "word" | "char" (per model card timestamp_levels)
    diarize: bool | None = None,                # label speakers (models with supports_diarization)
    lang: str | None = None,                    # ISO code; unset or "auto" = detect
    include_metadata: bool | None = None,       # URL sources only
    return_result_in_response: bool | None = None,
    webhook_url: str | None = None,
    webhook_secret: str | None = None,
) -> Job
```

`create_price()` takes the same options plus `platform` and `duration_seconds` (quote without a
source). Pass the same `ts_level` / `diarize` / `lang` / `include_metadata` as the job, otherwise
the quote is lower than the charge.

### `get_transcript()`

```python
transcript = client.transcription.get_transcript(
    result: JobResult,
    *,
    output_format: str | None = None,   # model card info.limits.output_format; "text" forces plain text
    alt_format: str | None = None,      # pick a rendition from results_alt_formats, e.g. "txt"
) -> Transcript
```

Uses the inline `result` if present, otherwise downloads `result_url` with a plain GET that sends
**no** Authorization header.

**Two result formats**, decided by the model card:

| `info.limits.output_format` | Example model | Result |
|-----------------------------|---------------|--------|
| `"json"` | `WhisperLargeV3Ct2` | Always a JSON document: `text`, `language`, `segments` (`start`, `end`, `speaker`, `words`) |
| `null` | `WhisperLargeV3` | Plain text; timestamps inline (`[0:50 - 1:49] ...`) when requested |

`deapi_sdk.parse_transcript(raw, output_format=None)` does the parsing without a client and never
raises. `deapi_sdk.transcript_output_format(model_card)` reads the card field.

```python
class Transcript(BaseModel):
    text: str
    format: Literal["json", "text"]
    language: str | None
    ts_level: str | None
    diarization_available: bool | None
    segments: list[TranscriptSegment]     # empty for format "text"
    raw: str                              # untouched result string

class TranscriptSegment(BaseModel):       # start, end, text, speaker, words
class TranscriptWord(BaseModel):          # word, start, end, speaker, score
```

---

## Resource: embeddings

```python
job = client.embeddings.create(
    *, input: str | list[str], model: str,
    return_result_in_response: bool | None = None,
    webhook_url: str | None = None, webhook_secret: str | None = None,
) -> Job
```

`create_price()`: `input`, `model`, `return_result_in_response`.

---

## Resource: ocr

```python
job = client.ocr.extract(
    *, image: FileInput, model: str,
    language: str | None = None, format: str | None = None,
    return_result_in_response: bool | None = None,
    webhook_url: str | None = None, webhook_secret: str | None = None,
) -> Job
```

`extract_price()`: `model`, `image` **or** `width` + `height`, optional `language`.

---

## Resource: prompts

Prompt methods return their result **immediately** (no polling).

### `enhance()`

```python
result = client.prompts.enhance(
    *,
    prompt: str,                         # min 3 chars
    type: str,
    model_slug: str,
    negative_prompt: str | None = None,
    image: FileInput | None = None,      # required for images.edits and videos.animations
) -> EnhancePromptResult
```

`type` is one of `images.generations`, `images.edits`, `images.upscales`,
`images.background-removals`, `images.ocr`, `videos.generations`, `videos.animations`,
`videos.upscales`, `videos.background-removals`, `audio.speech`, `audio.music`,
`audio.transcriptions`, `embeddings`. `enhance_price()` takes the same parameters.

---

## Resource: models

```python
response = client.models.list(
    *,
    limit: int | None = None,                  # page size (default 50, server cap 100)
    page: int = 1,
    inference_types: list[str] | None = None,  # e.g. ["txt2img", "img2img"]
    tags: list[str] | None = None,             # e.g. ["new", "beta"]
    sort: str | None = None,                   # id | name | created_at | updated_at, "-" prefix = descending
    per_page: int | None = None,               # deprecated alias for limit
) -> ModelsResponse

models = client.models.list_all(*, inference_types=None, tags=None, sort=None) -> list[ModelInfo]
model = client.models.get(slug: str) -> ModelInfo | None
```

`ModelInfo` has `name`, `slug`, `inference_types`, `status`, `tags`, `info` (with `limits`,
`features`, `defaults`), `loras`, `languages`. Helper properties: `limits`, `features`.

---

## Resource: jobs

Resume or inspect a job by id, for example after a `JobTimeoutError`.

```python
job = client.jobs.get(request_id)       # Job handle, no HTTP request
result = client.jobs.status(request_id) # poll once -> JobResult
result = client.jobs.get(request_id).wait()
```

---

## Resource: account

**Every call needs an ADMIN key**; an API key gets 403 `insufficient_permissions`
(`InsufficientPermissionsError`).

| Method | Returns | Endpoint |
|--------|---------|----------|
| `account.get()` | `Account` (`balance`, `currency`, `auto_top_up`) | `GET /account` |
| `account.balance()` | `Balance` | `GET /account/balance` |
| `account.usage(*, from_=None, to=None, group_by=None)` | `UsageReport` | `GET /account/usage` |
| `account.payments(*, limit=None, page=None, per_page=None)` | `PaymentsPage` | `GET /account/payments` |

`group_by` is `day`, `hour`, `key`, `type` or `model`. `from_` / `to` accept ISO strings, `date` or
`datetime`. `limit` defaults to 50 (max 100); `per_page` is an alias.

---

## Resource: keys

`current()` and `current_usage()` work with any key; everything else needs an ADMIN key.

| Method | Returns |
|--------|---------|
| `keys.current()` | `ApiKey` (the calling key) |
| `keys.current_usage(*, from_=None, to=None, group_by=None)` | `UsageReport` (`group_by`: day, hour, type, model) |
| `keys.list(*, limit=None, page=None, per_page=None)` | `ApiKeysPage` |
| `keys.get(key_id)` | `ApiKey` |
| `keys.usage(key_id, *, from_=None, to=None, group_by=None)` | `UsageReport` |
| `keys.create(*, name, expires_at=None, limit=None, reset_interval=None, ip_whitelist=None, role=None)` | `ApiKeyCreated` |
| `keys.update(key_id, *, active=..., expires_at=..., limit=..., reset_interval=..., ip_whitelist=...)` | `ApiKey` |
| `keys.delete(key_id)` | `None` |

- `create()` makes an API-role key. `reset_interval`: `none` (default), `daily`, `weekly`,
  `monthly`. `limit` is the spend cap (`None` = unlimited). `ip_whitelist` holds at most 20 entries.
  The returned `secret` is shown **once**; store it immediately.
- `update()` sends only the fields you pass; pass `None` to clear `expires_at` / `limit`.
  `active=False` disables the key (status `revoked`), `active=True` re-enables it.
- Creating beyond the account's cap raises `ActiveKeyLimitReachedError`.
- `list()` returns the account's API-role keys only; ADMIN keys (managed in the dashboard) are not listed.

---

## Job Polling

### `Job`

```python
class Job:
    request_id: str

    def status() -> JobResult           # poll once
    def wait(
        *,
        poll_interval: float = 1.0,
        max_wait: float = 300.0,
        backoff_factor: float = 1.5,
        max_interval: float = 10.0,
    ) -> JobResult                      # raises JobTimeoutError (has .request_id) on timeout
    def is_done() -> bool               # each makes an HTTP request
    def is_error() -> bool
```

A timeout does not cancel the job: resume with `client.jobs.get(request_id).wait()`. `AsyncJob`
has the same interface with `async` methods.

---

## Realtime (WebSocket)

Live job status over one WebSocket connection. Needs the optional extra:
`pip install "deapi-sdk[ws] @ git+https://github.com/deapi-ai/deapi-python-sdk.git"` (installs
`websockets` 17.x). Without it, connecting raises `ImportError`.

### `client.realtime()`

```python
rt = client.realtime(
    client_id: str | int | None = None,  # "Client ID" from the dashboard; falls back to DEAPI_CLIENT_ID
    *,
    ws_url: str | None = None,           # falls back to DEAPI_WS_URL (default "wss://soketi.deapi.ai")
    app_key: str | None = None,          # falls back to DEAPI_WS_APP_KEY (default: the public app key)
    connect_timeout: float = 30.0,
) -> Realtime                            # AsyncDeapiClient: AsyncRealtime
```

`client_id` is an opaque string. The environment variable is used only when the argument is not
passed; an explicit argument always wins. Servers handling many users should pass it per user.
`ws_url` and `app_key` are only needed for non-production environments.

### `Realtime` / `AsyncRealtime`

```python
class Realtime:
    def connect() -> None               # open the socket and subscribe to your channel
    def close() -> None
    connected: bool                     # property: is the socket currently up
    def wait(
        job_or_request_id: Job | str,
        *,
        on_progress: Callable[[JobStatusEvent], None] | None = None,
        max_wait: float = 300.0,
        poll_interval: float = 10.0,
    ) -> JobResult                      # raises JobTimeoutError (has .request_id) after max_wait
    def events() -> Iterator[JobStatusEvent]   # all status events on the channel; ends when closed
```

`with client.realtime(...) as rt:` connects on enter and closes on exit (`connect()` / `close()` do
the same explicitly). `AsyncRealtime` has the same interface with `async` methods: `async with`,
`await rt.connect()`, `await rt.close()`, `await rt.wait(...)` (`on_progress` may be a plain or async
function) and `async for event in rt.events()`.

`wait()` accepts a `Job` / `AsyncJob` or a request id:

- WebSocket events drive progress and completion. On `done` the SDK fetches the final `JobResult`
  over HTTP (fresh signed `result_url`, all fields).
- Failed jobs are **not** announced over WebSocket. `wait()` checks the job status over HTTP right
  away, every `poll_interval` seconds, and right after every reconnect. A failed job is returned
  with `status == "error"`, exactly like `Job.wait()`; it is not raised.
- Open the connection **before** submitting jobs so no event is missed. A job that finished earlier
  is still returned thanks to the immediate check.

The connection reconnects automatically with exponential backoff and re-subscribes. Events sent
while disconnected are lost; `wait()` compensates with the HTTP check after reconnecting.

---

## Types & Models

```python
class JobResult(BaseModel):
    status: JobStatus                       # "pending" | "processing" | "done" | "error"
    preview: str | None
    result_url: str | None
    results_alt_formats: dict | None        # {"jpg": "...", "webp": "..."}
    result: str | None                      # inline result (return_result_in_response=True)
    progress: float | None
    error_code: str | None                  # on failure: WORKER_TIMEOUT | WORKER_ERROR | PROCESSING_ERROR
    error_reason: str | None
    retryable: bool | None
    refunded: bool | None
    failed_at: str | None
    price: dict | float | None              # actual cost, e.g. {"amount": 0.01, "is_estimated": False}
    metadata: dict | None
    prompt_boosted: bool | None
    prompt_boost: PromptBoost | None

class JobStatusEvent(BaseModel):            # realtime status event
    request_id: str
    status: JobStatus | str                 # "processing" (repeated with growing progress) | "done"; unknown values kept as str
    progress: float | None                  # 0-100
    preview: str | None                     # base64 data URI preview (image jobs)
    result_url: str | None                  # only on "done"; signed, expires

class PriceResult(BaseModel):  price: float
class Balance(BaseModel):      balance: float
class ModelsResponse(BaseModel):  data: list[ModelInfo]; links; meta   # meta.current_page, last_page, per_page, total
class EnhancePromptResult(BaseModel):  prompt: str; negative_prompt: str | None
```

Account and key types: `Account`, `AutoTopUp`, `Payment`, `PaymentsPage`, `UsageReport`
(`totals`, `groups`, shortcuts `cost` / `requests` / `errors`), `ApiKey` (never contains the
secret), `ApiKeyCreated` (adds `secret`), `ApiKeysPage`; import from `deapi_sdk.types`.

```python
class JobStatus(StrEnum):  PENDING, PROCESSING, DONE, ERROR
class InferenceType(StrEnum):  TXT2IMG, IMG2IMG, TXT2AUDIO, TXT2MUSIC, ...   # see deapi_sdk._types
```

---

## Exceptions

Import everything from the top-level package: `from deapi_sdk import ...`.

```
DeapiError                              # .message  .status_code  .body  .code
├── AuthenticationError                 # 401  missing_key | invalid_key | key_expired | key_revoked
├── PaymentRequiredError                # 402
│   └── KeyLimitExceededError           # 402  key_limit_exceeded - the KEY's spend limit
├── PermissionDeniedError               # 403  unknown code
│   ├── IpNotAllowedError               # 403  ip_not_allowed
│   ├── InsufficientPermissionsError    # 403  insufficient_permissions
│   └── AccountSuspendedError           # 403  without a code
├── NotFoundError                       # 404
├── PayloadTooLargeError                # 413
├── ValidationError                     # 422  .errors
│   ├── InsufficientBalanceError        # 422  insufficient_balance - the ACCOUNT is out of credit
│   └── ActiveKeyLimitReachedError      # 422  active_key_limit_reached
├── RateLimitError                      # 429  .retry_after  .limit_type
├── ServerError                         # 5xx
├── NetworkError                        # connection failure / timeout
├── RealtimeError                       # fatal WebSocket connection / subscription failure; .close_code
└── JobTimeoutError                     # polling exceeded max_wait; .request_id
```

| Attribute | Meaning |
|-----------|---------|
| `code` | Machine-readable code from the API body (`None` if absent) |
| `status_code` | HTTP status (`None` for `NetworkError` / `JobTimeoutError`) |
| `message` | Human-readable message |
| `body` | Parsed response body, if any |

Specific attributes:

```python
class ValidationError(DeapiError):
    errors: dict[str, list[str]]        # {"field": ["message", ...]}

class KeyLimitExceededError(PaymentRequiredError):
    limit: float | None; used: float | None; remaining: float | None
    resets_at: str | None               # ISO datetime or None

class RateLimitError(DeapiError):
    retry_after: float                  # seconds (Retry-After header)
    limit_type: str                     # e.g. "minute" or "daily"

class JobTimeoutError(DeapiError):
    request_id: str | None              # resume: client.jobs.get(request_id).wait()

class RealtimeError(DeapiError):
    close_code: int | None              # WebSocket close code, when caused by one
```

`RealtimeError` covers fatal realtime failures, such as the channel authorization being refused
(a `client_id` that does not belong to this API key) or the server refusing the connection. HTTP
authentication failures during the channel authorization surface as the usual `AuthenticationError`
/ `PermissionDeniedError`.

`KeyLimitExceededError` (402) is the **key's** spend limit; `InsufficientBalanceError` (422) means
the **account** has no credit. They need different fixes (raise the key's limit vs. top up).

Backward compatibility: `except AccountSuspendedError` (it used to catch every 403) and
`except ValidationError` (still catches `InsufficientBalanceError`) keep working. Catch
`PermissionDeniedError` to handle every 403.

### Retry policy

Up to `max_retries` attempts with exponential backoff (429 waits for `Retry-After`).

| Request | Retried on |
|---------|-----------|
| `GET` (job status, models, account, keys) | 429, 5xx, network errors |
| Every other method: job-creating `POST` (price quotes are POSTs too), `PATCH`, `DELETE` | 429, or when the connection was never established |

Non-idempotent requests are **never** retried after a timeout or a 5xx, because the server may
already have accepted (and billed) the job. After a `NetworkError` / `ServerError` on submission,
check your jobs before resubmitting.

---

## Webhook Utilities

```python
from deapi_sdk.webhook import verify_signature, construct_event, InvalidSignatureError
```

### `verify_signature()`

Verify a webhook's HMAC-SHA256 signature; raises `InvalidSignatureError` on failure.

```python
verify_signature(
    *,
    payload: bytes | str,           # raw request body
    signature: str,                 # X-DeAPI-Signature header
    timestamp: str,                 # X-DeAPI-Timestamp header
    secret: str,                    # the webhook_secret you passed when creating the job
    tolerance: int = 300,           # max age in seconds (0 = disable)
) -> None
```

### `construct_event()`

Verify and parse into a `WebhookEvent`. Same parameters as `verify_signature()`.

```python
class WebhookEvent(BaseModel):
    event: str                      # "job.completed", "job.processing", "job.failed"
    delivery_id: str
    timestamp: str
    data: WebhookEventData
    type: str                       # property, alias for event

class WebhookEventData(BaseModel):
    job_request_id: str
    status: str
    previous_status: str | None
    job_type: str | None
    completed_at: str | None
    result_url: str | None
    processing_time_ms: int | None
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEAPI_API_KEY` | *(required)* | API key, `sk-dp-<id>-<secret>` |
| `DEAPI_BASE_URL` | `https://api.deapi.ai` | API base URL |
| `DEAPI_API_VERSION` | `v2` | API version (only `v2` is accepted) |
| `DEAPI_FILL_MODEL_DEFAULTS` | on | `0` stops the SDK filling unset parameters from model-card defaults |
| `DEAPI_CLIENT_ID` | *(none)* | Client ID for `client.realtime()` when `client_id` is not passed |
| `DEAPI_WS_URL` | `wss://soketi.deapi.ai` | WebSocket URL (non-production environments only) |
| `DEAPI_WS_APP_KEY` | public app key | WebSocket app key (non-production environments only) |
