# DeAPI Python SDK

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official Python SDK for the [DeAPI](https://deapi.ai) distributed AI inference platform. API details live in the [DeAPI docs](https://docs.deapi.ai).

Generate images, videos, audio, transcriptions, embeddings, and more — all through a clean, type-safe Python interface with sync and async support.

## Features

- **Image generation** — text-to-image, image edits (one or many sources), upscale, background removal
- **Video generation** — text-to-video, image-to-video (keyframes or reference images/videos/audio), audio-to-video, upscale
- **Video description** — timestamped events and scenes of a video
- **Audio** — text-to-speech (with voice cloning), text-to-music (with a style reference)
- **Transcription** — URLs and file uploads, plain-text or structured JSON transcripts
- **Embeddings, OCR, prompt enhancement**
- **Account and keys** — balance, usage, payments, API-key management (ADMIN keys)
- **Sync + Async** — both `DeapiClient` and `AsyncDeapiClient`
- **Job polling** — automatic exponential backoff with `.wait()`, resumable after a timeout
- **Real-time status** — live job progress over WebSocket (optional `[ws]` extra)
- **Precise errors** — every exception carries the API's machine-readable `code`
- **Safe auto-retry** — GETs retry on 429/5xx/network errors; job-creating POSTs never retry after a timeout or 5xx (no double charges)
- **Model-card defaults** — `width`, `height`, `steps`, `fps`, `seed`, ... are optional; the SDK fills them from the model card
- **Type-safe** — full type hints, Pydantic v2 models, `py.typed` marker
- **Webhook verification** — HMAC-SHA256 signature validation utility
- **File flexibility** — accepts file paths, `Path` objects, bytes, or file-like objects

## Installation

The SDK is **not on PyPI yet**. Install it straight from GitHub:

```bash
pip install git+https://github.com/deapi-ai/deapi-python-sdk.git
# or pin a branch, tag or commit:
pip install git+https://github.com/deapi-ai/deapi-python-sdk.git@main
```

Real-time job updates over WebSocket need the optional `ws` extra (installs `websockets`):

```bash
pip install "deapi-sdk[ws] @ git+https://github.com/deapi-ai/deapi-python-sdk.git"
```

> `pip install deapi` installs an **unrelated** package. Do not use it.

For development (`pytest` runs from the repo root):

```bash
git clone https://github.com/deapi-ai/deapi-python-sdk.git
cd deapi-python-sdk
pip install -e ".[dev]"
pytest
```

## API keys

Create a key in your [DeAPI dashboard](https://deapi.ai). Current keys look like
`sk-dp-<id>-<secret>` (older `<id>|<secret>` keys keep working). Keep them out of source control:
use an environment variable or a `.env` file (see [`.env.example`](.env.example)).

Keys have a role. **ADMIN** keys can do everything. **API** keys run jobs but get a 403
`insufficient_permissions` on balance, account and key-management endpoints
(`client.balance()`, `client.account`, and `client.keys` except `current()` / `current_usage()`).

## Quick Start

```python
from deapi_sdk import DeapiClient

client = DeapiClient(api_key="sk-dp-<id>-<secret>")

job = client.images.generate(
    prompt="a cat floating in a nebula, photorealistic",
    model="Flux1schnell",
    width=1024,
    height=1024,
    seed=42,
)

result = job.wait()          # polls with exponential backoff
print(result.result_url)
```

## Configuration

```python
client = DeapiClient(
    api_key="sk-dp-<id>-<secret>",              # or DEAPI_API_KEY
    base_url="https://api.deapi.ai",            # or DEAPI_BASE_URL
    timeout=60.0,                               # request timeout in seconds (default: 30)
    max_retries=5,                              # retry attempts (default: 3)
    api_version="v2",                           # only "v2" is accepted (default)
    fill_model_defaults=True,                   # default; see "Model-card defaults"
)
```

| Option | Environment variable | Default |
|--------|----------------------|---------|
| `api_key` | `DEAPI_API_KEY` | required |
| `base_url` | `DEAPI_BASE_URL` | `https://api.deapi.ai` |
| `api_version` | `DEAPI_API_VERSION` | `v2` |
| `fill_model_defaults` | `DEAPI_FILL_MODEL_DEFAULTS` (`0` disables) | on |
| `client_id` (`client.realtime()`) | `DEAPI_CLIENT_ID` | none |
| `ws_url` (`client.realtime()`) | `DEAPI_WS_URL` | `wss://soketi.deapi.ai` |
| `app_key` (`client.realtime()`) | `DEAPI_WS_APP_KEY` | public app key |
| `timeout` | — | `30.0` |
| `max_retries` | — | `3` |

```python
client = DeapiClient()                # reads the environment

with DeapiClient() as client:         # closes the HTTP connection on exit
    ...
```

The SDK targets deAPI API v2. `api_version` accepts only `"v2"`; see [Upgrading to 0.4.0](#upgrading-to-040).

### Model-card defaults

Parameters that depend on the model (`width`, `height`, `steps`, `frames`, `fps`, `seed`, `lang`,
`speed`, `format`, `sample_rate`, `inference_steps`, `guidance_scale`, ...) are optional. When you
leave one unset, the SDK fills it from the model's `info.defaults` in `GET /models` (one catalogue
request per client, then cached), and `seed` defaults to `-1` (random). Explicit values always win.
Disable with `DeapiClient(fill_model_defaults=False)` or `DEAPI_FILL_MODEL_DEFAULTS=0`.

## Usage

### Image Generation

```python
# Text-to-image
job = client.images.generate(
    prompt="mountain landscape at sunset",
    negative_prompt="blurry, low quality",
    model="Flux1schnell",
    width=1024,
    height=1024,
    seed=42,
    loras=[{"name": "detail-enhancer", "weight": 0.8}],
    quality="high",         # model tier, where the model supports it
    enhance_prompt=True,    # boost the prompt inside the job (fee billed on the job, not in *_price)
)
result = job.wait()
print(result.result_url)
print(result.results_alt_formats)  # {"jpg": "...", "webp": "..."}

# Image edits: image= for one source, images=[...] for several (not both)
job = client.images.transform(
    prompt="make it look like a watercolor painting",
    image="photo.jpg",                 # path, Path, bytes or file-like
    model="QwenImageEdit_Plus_NF4",
)

job = client.images.upscale(image="low_res.png", model="RealESRGAN_x4")
job = client.images.remove_background(image="photo.png", model="Ben2")
```

### Video Generation

```python
# Text-to-video
job = client.video.generate(
    prompt="a rocket launching into space",
    model="Ltx2_19B_Dist_FP8",
    width=768, height=512, frames=120, fps=24,
)

# Image-to-video with keyframes
job = client.video.animate(
    prompt="gentle camera pan across the scene",
    first_frame_image="landscape.jpg",
    model="Ltx2_19B_Dist_FP8",
)

# Reference mode: list order binds files to <Picture N> / <Video N> / <Audio N> in the prompt.
# Mutually exclusive with first/last frame images.
job = client.video.animate(
    prompt="<Picture 1> walks through the scene from <Video 1>, speaking <Audio 1>",
    model="<model-with-reference-support>",
    ref_images=["character.png"],
    ref_videos=["scene.mp4"],
    ref_audios=["voice.mp3"],
)

# Audio-to-video
job = client.video.generate_from_audio(
    prompt="visualize the music with abstract shapes",
    audio="track.mp3",
    model="ltx-audio2video",
)

# Describe a video: events, scenes and metadata
job = client.video.describe(
    model="Marlin_2B",
    video_url="https://youtube.com/watch?v=dQw4w9WgXcQ",
    preset="balanced",                 # fast | balanced | detail (models with presets)
)
description = client.video.get_description(job.wait())   # VideoDescription: events, scenes, meta
```

Reference limits (`max_ref_images`, `max_ref_videos`, `max_ref_audios`) come from the model card;
the API never accepts more than 9 images, 1 video or 3 audios. `video.replace()` and
`video.remove_background()` remain for compatibility, but no model currently serves them.

### Audio

```python
# Text-to-speech
job = client.audio.synthesize(
    text="Hello, welcome to DeAPI!",
    model="Kokoro",
    voice="af_sky",
    lang="en-us",
    format="mp3",
)

# Text-to-music, optionally with a style reference clip
job = client.audio.compose(
    caption="upbeat electronic track with synth leads",
    model="MusicGen",
    duration=30,
    reference_audio="style.mp3",
)
```

### Transcription

```python
job = client.transcription.create(
    source_url="https://youtube.com/watch?v=dQw4w9WgXcQ",   # or source_file="interview.mp3"
    model="WhisperLargeV3Ct2",
    ts_level="segment",       # segment | word | char (implies include_ts)
    diarize=True,             # label speakers (implies timestamps)
    lang="en",                # unset or "auto" = detect
    include_metadata=True,
)
transcript = client.transcription.get_transcript(job.wait())
print(transcript.format, transcript.text)
for seg in transcript.segments:
    print(seg.start, seg.end, seg.speaker, seg.text)
```

**Two transcript formats.** Models whose card has `info.limits.output_format: "json"` (for example
`WhisperLargeV3Ct2`) always return a JSON document (`text`, `language`, `segments` with
`start`/`end`/`speaker`/`words`). Models with `output_format: null` (for example `WhisperLargeV3`)
return plain text with timestamps inline. `client.transcription.get_transcript(result)` (or
`deapi_sdk.parse_transcript(raw)`) handles both and returns a `Transcript` whose `format` is `"json"` or
`"text"` (`segments` is empty for text). Result URLs are downloaded without your API key.

### Embeddings and OCR

```python
job = client.embeddings.create(
    input=["first sentence", "second sentence"],   # or a single string
    model="Bge_M3_FP16",
    return_result_in_response=True,
)

job = client.ocr.extract(image="scan.png", model="<ocr-model>", return_result_in_response=True)
```

### Prompt Enhancement

Prompt methods return results immediately (no job polling):

```python
enhanced = client.prompts.enhance(
    prompt="cat in space",
    type="images.generations",   # images.edits, videos.generations, videos.animations, audio.speech, ...
    model_slug="Flux1schnell",
    negative_prompt="blurry",
)
print(enhanced.prompt)
```

`images.edits` and `videos.animations` also need `image=`. The per-type methods
`enhance_image()`, `enhance_video()` etc. were removed in 0.4.0 (see [Upgrading to 0.4.0](#upgrading-to-040)).

### Price Calculation

Every job method has a `_price` counterpart. The `enhance_prompt` fee is billed on the job and is
not part of the quote.

```python
price = client.images.generate_price(prompt="a cat in space", model="Flux1schnell", width=1024, height=1024)
print(f"Cost: ${price.price}")
```

### Account, Keys and Balance

```python
client.balance().balance                        # ADMIN key only

client.account.get()                            # balance, currency, auto top-up (ADMIN)
client.account.usage(group_by="day")            # from_=, to=; group_by day|hour|key|type|model
client.account.payments(limit=20)

client.keys.current()                           # the calling key (any role)
client.keys.current_usage(group_by="model")     # usage of the calling key (any role)

# ADMIN only
client.keys.list(limit=50)
created = client.keys.create(name="ci", limit=25.0, reset_interval="monthly")
print(created.secret)                           # shown ONCE, store it now
client.keys.update(created.id, active=False)    # disable; active=True re-enables
client.keys.delete(created.id)
```

### Models

```python
page = client.models.list(limit=50, page=1, inference_types=["txt2img"], tags=["new"], sort="-created_at")
everything = client.models.list_all()
model = client.models.get("Flux1schnell")       # ModelInfo or None
```

`per_page` is a deprecated alias for `limit`.

## Async Usage

Every method has an async equivalent via `AsyncDeapiClient`:

```python
import asyncio
from deapi_sdk import AsyncDeapiClient

async def main():
    async with AsyncDeapiClient(api_key="sk-dp-<id>-<secret>") as client:
        jobs = await asyncio.gather(*[
            client.images.generate(prompt=p, model="Flux1schnell", width=1024, height=1024)
            for p in ("mountain landscape", "ocean sunset", "forest path")
        ])
        results = await asyncio.gather(*[job.wait() for job in jobs])
        for result in results:
            print(result.result_url)

asyncio.run(main())
```

## Job Polling

All job methods return a `Job` (or `AsyncJob`):

```python
job = client.images.generate(...)

result = job.wait(
    poll_interval=1.0,      # initial interval between polls (default: 1s)
    max_wait=300.0,         # timeout in seconds (default: 5 min)
    backoff_factor=1.5,
    max_interval=10.0,
)

result = job.status()       # or poll once yourself: result.status, result.progress
```

If `wait()` times out it raises `JobTimeoutError`, but **the job keeps running**. Resume it any time:

```python
from deapi_sdk import JobTimeoutError

try:
    result = job.wait(max_wait=60)
except JobTimeoutError as e:
    result = client.jobs.get(e.request_id).wait()   # or client.jobs.status(e.request_id)
```

## Real-time Updates (WebSocket)

Instead of polling, follow your jobs live over one WebSocket connection. Install the `ws` extra
(see [Installation](#installation)) and open a connection **before** submitting jobs:

```python
from deapi_sdk import DeapiClient

with DeapiClient(api_key="sk-dp-<id>-<secret>") as client:
    with client.realtime(client_id="your-client-id") as rt:
        job = client.images.generate(prompt="a cat in space", model="Flux1schnell", width=1024, height=1024)

        result = rt.wait(job, on_progress=lambda e: print(e.status, e.progress))   # same JobResult as job.wait()
        print(result.status, result.result_url)

        for event in rt.events():      # every status event on your channel (all your jobs); blocks
            print(event.request_id, event.status, event.progress)
```

Async works the same way:

```python
async with AsyncDeapiClient() as client:
    async with client.realtime(client_id="your-client-id") as rt:
        job = await client.images.generate(prompt="a cat in space", model="Flux1schnell", width=1024, height=1024)
        result = await rt.wait(job, on_progress=callback)   # plain function or async function
        async for event in rt.events():
            ...
```

- **`client_id`** is the "Client ID" shown in your deAPI dashboard. It falls back to `DEAPI_CLIENT_ID`
  only when you do not pass it; an explicit argument always wins. Servers handling many users should
  pass it explicitly per user. A `client_id` that does not belong to the API key raises `RealtimeError`.
- **`rt.wait(job_or_request_id)`** accepts a `Job` or a request id. WebSocket events drive progress and
  completion; on `done` the SDK fetches the final `JobResult` over HTTP (fresh signed `result_url`, all
  fields).
- **Failed jobs are not announced over WebSocket.** `wait()` therefore also checks the job over HTTP
  every `poll_interval` seconds (default 10) and right after every reconnect. A failed job is returned
  with `status == "error"`, exactly like `job.wait()`. After `max_wait` (default 300 s) it raises
  `JobTimeoutError`.
- **Reconnects** happen automatically with exponential backoff and the channel is re-subscribed. Events
  sent while disconnected are lost; `wait()` compensates by checking the job over HTTP after
  reconnecting. `rt.connected` tells you whether the socket is up.
- `JobStatusEvent` has `request_id`, `status` (`processing`, repeated while `progress` grows, then `done`), `progress` (0-100),
  `preview` (base64 data URI, image jobs) and `result_url` (only on `done`).
- `ws_url` / `app_key` (env `DEAPI_WS_URL` / `DEAPI_WS_APP_KEY`) are only needed for non-production
  environments. Without the extra, connecting raises an `ImportError` telling you to install
  `deapi-sdk[ws]`.

See [`examples/realtime_status.py`](examples/realtime_status.py).

## Error Handling

Every exception derives from `DeapiError` and carries `.code` (the API's machine-readable code, or
`None`), `.status_code`, `.message` and `.body`.

```
DeapiError
├── AuthenticationError              401  missing_key | invalid_key | key_expired | key_revoked
├── PaymentRequiredError             402
│   └── KeyLimitExceededError        402  key_limit_exceeded: this KEY's spend limit
│                                         (limit, used, remaining, resets_at)
├── PermissionDeniedError            403  any other code
│   ├── IpNotAllowedError            403  ip_not_allowed
│   ├── InsufficientPermissionsError 403  insufficient_permissions (API key on an ADMIN endpoint)
│   └── AccountSuspendedError        403  no code
├── NotFoundError                    404
├── PayloadTooLargeError             413
├── ValidationError                  422  .errors = {"field": ["message", ...]}
│   ├── InsufficientBalanceError     422  insufficient_balance: the ACCOUNT is out of credit
│   └── ActiveKeyLimitReachedError   422  active_key_limit_reached
├── RateLimitError                   429  retry_after, limit_type
├── ServerError                      5xx
├── NetworkError                     connection failure / timeout
├── RealtimeError                    fatal WebSocket failure (.close_code)
└── JobTimeoutError                  polling exceeded max_wait (request_id; the job keeps running)
```

```python
from deapi_sdk import (
    DeapiError, InsufficientBalanceError, KeyLimitExceededError, IpNotAllowedError,
    InsufficientPermissionsError, ValidationError, RateLimitError, JobTimeoutError,
)

try:
    result = client.images.generate(prompt="a cat", model="Flux1schnell").wait()
except InsufficientBalanceError:
    print("Account out of credit, top up at https://deapi.ai")
except KeyLimitExceededError as e:
    print(f"This key's limit is used up: {e.used}/{e.limit}, resets {e.resets_at}")
except IpNotAllowedError:
    print("Your IP is not on the key's allowlist")
except InsufficientPermissionsError:
    print("Use an ADMIN key for this endpoint")
except ValidationError as e:
    print(e.message, e.errors)
except RateLimitError as e:
    print(f"Rate limited, retry in {e.retry_after}s ({e.limit_type})")
except JobTimeoutError as e:
    result = client.jobs.get(e.request_id).wait()
except DeapiError as e:
    print(f"{e.status_code} {e.code}: {e.message}")
```

Existing `except AccountSuspendedError` and `except ValidationError` handlers keep working.

**Retries** (up to `max_retries`):

- `GET` requests are retried on 429, 5xx and network errors.
- Job-creating `POST`s (and `PATCH` / `DELETE`) are retried **only** on 429 or when the connection
  was never established. They are never retried after a timeout or a 5xx, so one job is never paid
  for twice. After such an error, check whether the job exists before resubmitting.

## Webhooks

Every job method accepts `webhook_url` (must be `https`) and `webhook_secret` (32-255 characters,
requires `webhook_url`), a per-job HMAC secret used to sign that job's deliveries:

```python
import secrets

WEBHOOK_SECRET = secrets.token_urlsafe(48)

job = client.images.generate(
    prompt="a cat", model="Flux1schnell",
    webhook_url="https://example.com/webhooks/deapi",
    webhook_secret=WEBHOOK_SECRET,
)
```

Verify incoming requests with the same secret:

```python
from deapi_sdk.webhook import construct_event, InvalidSignatureError

@app.post("/webhooks/deapi")
async def handle_webhook(request: Request):
    payload = await request.body()

    try:
        event = construct_event(
            payload=payload,
            signature=request.headers["X-DeAPI-Signature"],
            timestamp=request.headers["X-DeAPI-Timestamp"],
            secret=WEBHOOK_SECRET,
            tolerance=300,  # reject events older than 5 minutes
        )
    except InvalidSignatureError:
        return {"error": "Invalid signature"}, 403

    if event.type == "job.completed":
        print(f"Job {event.data.job_request_id} completed: {event.data.result_url}")
    elif event.type == "job.failed":
        print(f"Job {event.data.job_request_id} failed")

    return {"ok": True}
```

## Upgrading to 0.4.0

0.4.0 supports only deAPI API v2 and removes everything that existed for API v1. `api_version` /
`DEAPI_API_VERSION` still exist but accept only `"v2"` (the default); any other value raises `ValueError`.

| Removed | Use instead |
|---------|-------------|
| `transcription.from_video_url()`, `from_audio_url()`, `from_audio_file()`, `from_video_file()` (and `*_price`) | `transcription.create(source_url=... \| source_file=..., ...)` / `create_price(...)` |
| `prompts.enhance_image()`, `enhance_video()`, `enhance_speech()`, `enhance_image2image()` (and `*_price`) | `prompts.enhance(prompt=..., type=..., model_slug=...)` / `enhance_price(...)` |
| `prompts.samples()` / `samples_price()` | no replacement (already raised `NotImplementedError`) |
| Types `VideoToTextParams`, `AudioToTextParams`, `AudioFileToTextParams`, `VideoFileToTextParams`, `EnhanceSpeechPromptResult`, `SamplePromptResult` | no replacement |
| Modules `deapi_sdk.resources.v1` / `.v2`, `deapi_sdk.types.v1` / `.v2` | `deapi_sdk.resources`, `deapi_sdk.types` (top-level `from deapi_sdk import ...` names are unchanged) |

`client.account` and `client.keys` are always available.

## API Reference

See [docs/api-reference.md](docs/api-reference.md) for method signatures, parameters and return
types, and [docs.deapi.ai](https://docs.deapi.ai) for the HTTP API.

## Examples

The [`examples/`](examples/) directory contains runnable scripts:

| File | Description |
|------|-------------|
| [`basic_image.py`](examples/basic_image.py) | Image generation, transformation, upscale |
| [`async_batch.py`](examples/async_batch.py) | Concurrent async job submission |
| [`transcription.py`](examples/transcription.py) | Transcription with JSON and plain-text transcripts |
| [`tts_and_audio.py`](examples/tts_and_audio.py) | Text-to-speech and music generation |
| [`video_generation.py`](examples/video_generation.py) | Text-to-video and image-to-video |
| [`realtime_status.py`](examples/realtime_status.py) | Live job progress over WebSocket |
| [`webhook_server.py`](examples/webhook_server.py) | Flask webhook receiver with signature verification |
| [`error_handling.py`](examples/error_handling.py) | Error handling patterns and best practices |

## Requirements

- Python 3.11+
- [`httpx`](https://www.python-httpx.org/) >= 0.25.0
- [`pydantic`](https://docs.pydantic.dev/) >= 2.6.0
- [`websockets`](https://websockets.readthedocs.io/) 17.x (optional, `ws` extra)

## License

MIT — see [LICENSE](LICENSE) for details.
