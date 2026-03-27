# DeAPI Python SDK

[![PyPI version](https://img.shields.io/pypi/v/deapi-python-sdk.svg)](https://pypi.org/project/deapi-python-sdk/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official Python SDK for the [DeAPI](https://deapi.ai) distributed AI inference platform.

Generate images, videos, audio, transcriptions, embeddings, and more — all through a clean, type-safe Python interface with sync and async support.

## Features

- **Image generation** — text-to-image, image-to-image, upscale, background removal
- **Video generation** — text-to-video, image-to-video, audio-to-video, video replace, upscale, background removal
- **Audio** — text-to-speech (with voice cloning), text-to-music
- **Transcription** — video URLs, audio URLs, file uploads, unified endpoint
- **Embeddings** — text-to-embedding (single or batch)
- **Prompt enhancement** — AI-powered prompt boosting for images, video, speech
- **OCR** — image-to-text extraction
- **Sync + Async** — both `DeapiClient` and `AsyncDeapiClient`
- **Job polling** — automatic exponential backoff with `.wait()`
- **Auto-retry** — built-in retry on rate limits (429) and server errors (5xx)
- **Type-safe** — full type hints, Pydantic v2 models, `py.typed` marker
- **Webhook verification** — HMAC-SHA256 signature validation utility
- **File flexibility** — accepts file paths, `Path` objects, bytes, or file-like objects

## Installation

```bash
pip install deapi-python-sdk
```

## Quick Start

```python
from deapi import DeapiClient

client = DeapiClient(api_key="sk-your-api-key")

# Generate an image
job = client.images.generate(
    prompt="a cat floating in a nebula, photorealistic",
    model="Flux1schnell",
    width=1024,
    height=1024,
    seed=42,
)

# Wait for the result (polls with exponential backoff)
result = job.wait()
print(result.result_url)
```

## Configuration

```python
client = DeapiClient(
    api_key="sk-...",                           # or set DEAPI_API_KEY env var
    base_url="https://api.deapi.ai",            # or set DEAPI_BASE_URL env var
    timeout=60.0,                               # request timeout in seconds (default: 30)
    max_retries=5,                              # retry attempts for 429/5xx (default: 3)
    api_version="v1",                           # or set DEAPI_API_VERSION env var
)
```

**Environment variables** — if you prefer not to pass credentials in code:

```bash
export DEAPI_API_KEY="sk-your-key"
export DEAPI_BASE_URL="https://api.deapi.ai"    # optional
export DEAPI_API_VERSION="v1"                    # optional
```

```python
client = DeapiClient()  # reads from env vars
```

**Context manager** — automatically closes the HTTP connection:

```python
with DeapiClient(api_key="sk-...") as client:
    result = client.images.generate(...).wait()
```

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
)
result = job.wait()
print(result.result_url)
print(result.results_alt_formats)  # {"jpg": "...", "webp": "..."}

# Image-to-image (accepts path string, Path, bytes, or file-like)
job = client.images.transform(
    prompt="make it look like a watercolor painting",
    image="photo.jpg",
    model="QwenImageEdit_Plus_NF4",
    steps=30,
    seed=42,
)

# Image upscale
job = client.images.upscale(image="low_res.png", model="RealESRGAN_x4")

# Background removal
job = client.images.remove_background(image="photo.png", model="Ben2")
```

### Video Generation

```python
# Text-to-video
job = client.video.generate(
    prompt="a rocket launching into space",
    model="Ltx2_19B_Dist_FP8",
    width=768,
    height=512,
    steps=1,
    seed=42,
    frames=120,
    fps=24,
)

# Image-to-video (animate a still image)
job = client.video.animate(
    prompt="gentle camera pan across the scene",
    first_frame_image="landscape.jpg",
    model="Ltx2_19B_Dist_FP8",
    seed=42,
    width=768,
    height=512,
    frames=120,
    fps=24,
)

# Audio-to-video (sync visuals to audio)
job = client.video.generate_from_audio(
    prompt="visualize the music with abstract shapes",
    audio="track.mp3",
    model="ltx-audio2video",
    width=512,
    height=512,
    seed=42,
    frames=97,
    fps=24,
)

# Video replace (swap character using a reference image)
job = client.video.replace(
    video="input.mp4",
    ref_image="reference_face.png",
    model="vid-replace-model",
    prompt="replace the character",
)
```

### Audio

```python
# Text-to-speech
job = client.audio.synthesize(
    text="Hello, welcome to DeAPI!",
    model="Kokoro",
    voice="af_sky",
    lang="en-us",
    format="mp3",
    speed=1.0,
    sample_rate=24000,
)

# Text-to-music
job = client.audio.compose(
    caption="upbeat electronic track with synth leads",
    model="MusicGen",
    duration=30,
    inference_steps=50,
    guidance_scale=3.0,
    seed=42,
    format="mp3",
)
```

### Transcription

```python
# Unified endpoint (recommended) — auto-detects source type
job = client.transcription.create(
    source_url="https://youtube.com/watch?v=dQw4w9WgXcQ",
    include_ts=True,
    model="WhisperLargeV3",
    return_result_in_response=True,
)
result = job.wait()
print(result.result)  # inline transcription text

# File upload
job = client.transcription.create(
    source_file="interview.mp3",
    include_ts=False,
    model="WhisperLargeV3",
)

# Legacy endpoints also available:
# client.transcription.from_video_url(...)
# client.transcription.from_audio_url(...)   (Twitter Spaces)
# client.transcription.from_audio_file(...)
# client.transcription.from_video_file(...)
```

### Embeddings

```python
# Single text
job = client.embeddings.create(
    input="The quick brown fox",
    model="Bge_M3_FP16",
    return_result_in_response=True,
)

# Batch
job = client.embeddings.create(
    input=["first sentence", "second sentence", "third sentence"],
    model="Bge_M3_FP16",
    return_result_in_response=True,
)
```

### Prompt Enhancement

Prompt methods return results immediately (no job polling needed):

```python
# Enhance an image prompt
enhanced = client.prompts.enhance_image(
    prompt="cat in space",
    negative_prompt="blurry",
)
print(enhanced.prompt)            # AI-enhanced prompt
print(enhanced.negative_prompt)   # AI-enhanced negative prompt

# Also available:
# client.prompts.enhance_video(prompt=..., image=...)
# client.prompts.enhance_speech(prompt=..., lang_code=...)
# client.prompts.enhance_image2image(prompt=..., image=...)

# Get sample prompts for inspiration
sample = client.prompts.samples(type="image", topic="nature")
print(sample.prompt)
```

### Price Calculation

Every resource method has a `_price` counterpart to check cost before submitting:

```python
price = client.images.generate_price(
    prompt="a cat in space",
    model="Flux1schnell",
    width=1024,
    height=1024,
    seed=42,
)
print(f"Cost: ${price.price}")
```

### Account Balance

```python
balance = client.balance()
print(f"Credits: {balance.balance}")
```

### List Available Models

```python
models = client.models.list(per_page=50)
for model in models.data:
    print(f"{model.name}: {model.inference_types}")

# Filter by capability
image_models = client.models.list(inference_types=["txt2img"])
```

## Async Usage

Every method has an async equivalent via `AsyncDeapiClient`:

```python
import asyncio
from deapi import AsyncDeapiClient

async def main():
    async with AsyncDeapiClient(api_key="sk-...") as client:
        # Submit multiple jobs concurrently
        jobs = await asyncio.gather(
            client.images.generate(
                prompt="mountain landscape", model="Flux1schnell",
                width=1024, height=1024, seed=1,
            ),
            client.images.generate(
                prompt="ocean sunset", model="Flux1schnell",
                width=1024, height=1024, seed=2,
            ),
            client.images.generate(
                prompt="forest path", model="Flux1schnell",
                width=1024, height=1024, seed=3,
            ),
        )

        # Wait for all results concurrently
        results = await asyncio.gather(*[job.wait() for job in jobs])
        for i, result in enumerate(results):
            print(f"Job {i + 1}: {result.result_url}")

asyncio.run(main())
```

## Job Polling

All generation methods return a `Job` (or `AsyncJob`) object:

```python
job = client.images.generate(...)

# Option 1: Automatic polling with exponential backoff
result = job.wait(
    poll_interval=1.0,      # initial interval between polls (default: 1s)
    max_wait=300.0,         # timeout in seconds (default: 5 min)
    backoff_factor=1.5,     # backoff multiplier (default: 1.5x)
    max_interval=10.0,      # max seconds between polls (default: 10s)
)

# Option 2: Manual polling
import time

while True:
    result = job.status()
    print(f"Status: {result.status}, Progress: {result.progress}%")

    if result.status == "done":
        print(f"Result: {result.result_url}")
        break
    elif result.status == "error":
        print("Job failed!")
        break

    time.sleep(2)
```

## Error Handling

The SDK provides a detailed exception hierarchy:

```python
from deapi import DeapiClient
from deapi._exceptions import (
    DeapiError,                # Base exception
    AuthenticationError,       # 401 — invalid/missing API key
    AccountSuspendedError,     # 403 — account suspended
    NotFoundError,             # 404 — resource not found
    ValidationError,           # 422 — invalid parameters
    InsufficientBalanceError,  # 422 — not enough credits
    RateLimitError,            # 429 — rate limited
    ServerError,               # 5xx — server error
    JobTimeoutError,           # polling exceeded max_wait
    NetworkError,              # connection failure
)

try:
    job = client.images.generate(...)
    result = job.wait()
except InsufficientBalanceError:
    print("Not enough credits! Top up at https://deapi.ai")
except ValidationError as e:
    print(f"Invalid request: {e.message}")
    print(f"Field errors: {e.errors}")  # {"model": ["..."], "width": ["..."]}
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s (type: {e.limit_type})")
except JobTimeoutError:
    print("Job took too long — try increasing max_wait")
except NetworkError:
    print("Connection failed — check your network")
except DeapiError as e:
    print(f"API error {e.status_code}: {e.message}")
```

**Auto-retry** is built in — the SDK automatically retries on:
- **429 (rate limit)** — uses `Retry-After` header
- **5xx (server errors)** — exponential backoff
- **Network errors** — single retry

Configure with `max_retries` on the client constructor.

## Webhook Verification

If you use webhook URLs for job notifications, verify incoming requests:

```python
from deapi.webhook import construct_event, InvalidSignatureError

# In your Flask/FastAPI webhook handler:
@app.post("/webhooks/deapi")
async def handle_webhook(request: Request):
    payload = await request.body()

    try:
        event = construct_event(
            payload=payload,
            signature=request.headers["X-DeAPI-Signature"],
            timestamp=request.headers["X-DeAPI-Timestamp"],
            secret="your-webhook-secret",
            tolerance=300,  # reject events older than 5 minutes
        )
    except InvalidSignatureError:
        return {"error": "Invalid signature"}, 403

    if event.type == "job.completed":
        print(f"Job {event.data.job_request_id} completed!")
        print(f"Result: {event.data.result_url}")
    elif event.type == "job.failed":
        print(f"Job {event.data.job_request_id} failed")

    return {"ok": True}
```

## API Reference

See [docs/api-reference.md](docs/api-reference.md) for complete method signatures, parameter details, and return types for every resource.

## Examples

The [`examples/`](examples/) directory contains runnable scripts:

| File | Description |
|------|-------------|
| [`basic_image.py`](examples/basic_image.py) | Image generation, transformation, upscale |
| [`async_batch.py`](examples/async_batch.py) | Concurrent async job submission |
| [`transcription.py`](examples/transcription.py) | URL and file transcription |
| [`tts_and_audio.py`](examples/tts_and_audio.py) | Text-to-speech and music generation |
| [`video_generation.py`](examples/video_generation.py) | Text-to-video and image-to-video |
| [`webhook_server.py`](examples/webhook_server.py) | Flask webhook receiver with signature verification |
| [`error_handling.py`](examples/error_handling.py) | Error handling patterns and best practices |

## Requirements

- Python 3.9+
- [`httpx`](https://www.python-httpx.org/) >= 0.25.0
- [`pydantic`](https://docs.pydantic.dev/) >= 2.0.0

## License

MIT — see [LICENSE](LICENSE) for details.