"""Tests for v2 API support.

Verifies that:
* v2 endpoint paths get hit (renames mapped correctly).
* Unified endpoints work (transcription, prompt enhancer).
* Unified endpoints are the only route (no per-type legacy shims).
* The polling status URL uses ``/api/v2/jobs/{id}``.
* The default api_version is ``v2``.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from deapi_sdk import AsyncDeapiClient, DeapiClient

TEST_BASE_URL = "https://test.deapi.ai"
TEST_API_KEY = "test-key-123"

V2 = f"{TEST_BASE_URL}/api/v2"

JOB_RESPONSE = {"data": {"request_id": "v2-job-123"}}
PRICE_RESPONSE = {"data": {"price": 0.05}}
ENHANCE_RESPONSE = {"prompt": "enhanced", "negative_prompt": "no blur"}


# --- Defaults ---


class TestDefaults:
    def test_default_version_is_v2(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL)
        assert client._config.api_version == "v2"
        assert client._config.api_prefix == "/api/v2"
        client.close()

# --- Path renames (v2 routes get hit) ---


class TestV2Paths:
    @respx.mock
    def test_images_generate_hits_v2_path(self, client: DeapiClient) -> None:
        respx.post(f"{V2}/images/generations").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client.images.generate(
            prompt="cat in space", model="sdxl", width=512, height=512, seed=1,
        )
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_images_edit_hits_v2_path(self, client: DeapiClient) -> None:
        respx.post(f"{V2}/images/edits").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client.images.transform(
            prompt="paint blue", model="sdxl-edit", image=b"fake-image",
            steps=4, seed=1,
        )
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_images_upscale_hits_v2_path(self, client: DeapiClient) -> None:
        respx.post(f"{V2}/images/upscales").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client.images.upscale(image=b"fake-image", model="real-esrgan")
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_images_rmbg_hits_v2_path(self, client: DeapiClient) -> None:
        respx.post(f"{V2}/images/background-removals").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client.images.remove_background(image=b"fake-image", model="rembg")
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_video_generate_hits_v2_path(self, client: DeapiClient) -> None:
        respx.post(f"{V2}/videos/generations").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client.video.generate(
            prompt="ocean", model="wan2.2", width=512, height=512,
            steps=4, seed=1, frames=24, fps=24,
        )
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_video_animate_hits_v2_path(self, client: DeapiClient) -> None:
        respx.post(f"{V2}/videos/animations").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client.video.animate(
            prompt="zoom in", model="wan-i2v", first_frame_image=b"img",
            seed=1, width=512, height=512, frames=24, fps=24,
        )
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_video_upscale_hits_v2_path(self, client: DeapiClient) -> None:
        respx.post(f"{V2}/videos/upscales").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client.video.upscale(video=b"vid", model="real-esrgan")
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_video_rmbg_hits_v2_path(self, client: DeapiClient) -> None:
        respx.post(f"{V2}/videos/background-removals").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client.video.remove_background(video=b"vid", model="rembg")
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_video_replace_hits_v2_path(self, client: DeapiClient) -> None:
        respx.post(f"{V2}/videos/replacements").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client.video.replace(video=b"vid", ref_image=b"ref", model="wan-replace")
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_video_audio_sync_hits_v2_path(self, client: DeapiClient) -> None:
        respx.post(f"{V2}/videos/audio-syncs").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client.video.generate_from_audio(
            prompt="dance", audio=b"audio", model="wan-a2v",
            width=512, height=512, seed=1, frames=24, fps=24,
        )
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_audio_speech_hits_v2_path(self, client: DeapiClient) -> None:
        respx.post(f"{V2}/audio/speech").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client.audio.synthesize(
            text="hello world", model="kokoro", lang="en", format="wav",
            speed=1.0, sample_rate=24000.0, voice="alpha",
        )
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_audio_music_hits_v2_path(self, client: DeapiClient) -> None:
        respx.post(f"{V2}/audio/music").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client.audio.compose(
            caption="jazz beat", model="musicgen", duration=30.0,
            inference_steps=50, guidance_scale=3.0, seed=1, format="wav",
        )
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_embeddings_hits_v2_path(self, client: DeapiClient) -> None:
        respx.post(f"{V2}/embeddings").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client.embeddings.create(input="hello", model="bge-m3")
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_ocr_hits_v2_path(self, client: DeapiClient) -> None:
        respx.post(f"{V2}/images/ocr").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client.ocr.extract(image=b"img", model="paddleocr")
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_balance_hits_v2_path(self, client: DeapiClient) -> None:
        respx.get(f"{V2}/account/balance").mock(
            return_value=httpx.Response(200, json={"data": {"balance": 42.0}})
        )
        bal = client.balance()
        assert bal.balance == 42.0

    @respx.mock
    def test_models_hits_v2_path(self, client: DeapiClient) -> None:
        respx.get(f"{V2}/models").mock(
            return_value=httpx.Response(200, json={
                "data": [], "links": {}, "meta": {"current_page": 1, "last_page": 1, "per_page": 15, "total": 0},
            })
        )
        result = client.models.list()
        assert result.meta.total == 0


# --- Job polling URL ---


class TestV2Polling:
    @respx.mock
    def test_polling_uses_jobs_path(self, client: DeapiClient) -> None:
        respx.post(f"{V2}/images/generations").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        respx.get(f"{V2}/jobs/v2-job-123").mock(
            return_value=httpx.Response(200, json={
                "data": {"status": "done", "result_url": "https://cdn/image.png", "progress": 1.0},
            })
        )
        job = client.images.generate(
            prompt="cat", model="sdxl", width=512, height=512, seed=1,
        )
        result = job.status()
        assert result.result_url == "https://cdn/image.png"


# --- Unified transcription ---


class TestV2Transcription:
    @respx.mock
    def test_create_with_url(self, client: DeapiClient) -> None:
        route = respx.post(f"{V2}/audio/transcriptions").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client.transcription.create(
            source_url="https://example.com/a.mp4", include_ts=True, model="whisper",
        )
        assert job.request_id == "v2-job-123"
        body = json.loads(route.calls.last.request.content)
        assert body["source_url"] == "https://example.com/a.mp4"

    @respx.mock
    def test_create_with_file_uses_multipart(self, client: DeapiClient) -> None:
        route = respx.post(f"{V2}/audio/transcriptions").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        client.transcription.create(
            source_file=b"audio-bytes", include_ts=False, model="whisper",
        )
        ct = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in ct

    @respx.mock
    def test_create_price(self, client: DeapiClient) -> None:
        respx.post(f"{V2}/audio/transcriptions/price").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.transcription.create_price(
            include_ts=True, model="whisper", duration_seconds=60.0,
        )
        assert price.price == 0.05

# --- Unified prompt enhancer ---


class TestV2Prompts:
    @respx.mock
    def test_enhance_unified(self, client: DeapiClient) -> None:
        route = respx.post(f"{V2}/prompts/enhancements").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        result = client.prompts.enhance(
            prompt="a cat", type="images.generations", model_slug="sdxl",
        )
        assert result.prompt == "enhanced"
        assert result.negative_prompt == "no blur"
        # OpenAPI declares multipart/form-data only — must force multipart even
        # when no image is sent.
        ct = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in ct
        body = route.calls.last.request.content
        assert b"images.generations" in body
        assert b"sdxl" in body

    @respx.mock
    def test_enhance_with_image(self, client: DeapiClient) -> None:
        route = respx.post(f"{V2}/prompts/enhancements").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        client.prompts.enhance(
            prompt="a cat", type="images.edits", model_slug="sdxl-edit", image=b"img",
        )
        # Multipart should contain the image as a file part
        body = route.calls.last.request.content
        assert b"image" in body

    @respx.mock
    def test_enhance_price(self, client: DeapiClient) -> None:
        respx.post(f"{V2}/prompts/enhancements/price").mock(
            return_value=httpx.Response(200, json={"data": {"price": 0.001}})
        )
        price = client.prompts.enhance_price(
            prompt="a cat", type="images.generations", model_slug="sdxl",
        )
        assert price.price == 0.001

    @pytest.mark.parametrize(
        ("type_", "image"),
        [
            ("audio.speech", None),
            ("audio.music", None),
            ("videos.generations", None),
            ("videos.animations", b"img"),
            ("images.edits", b"img"),
        ],
    )
    @respx.mock
    def test_enhance_sends_type_and_optional_image(self, client: DeapiClient, type_: str, image: bytes | None) -> None:
        route = respx.post(f"{V2}/prompts/enhancements").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        result = client.prompts.enhance(prompt="a cat", type=type_, model_slug="some-model", image=image)
        assert result.prompt == "enhanced"
        body = route.calls.last.request.content
        assert type_.encode() in body
        assert b"some-model" in body
        assert (b'name="image"' in body) is (image is not None)


# --- Async parity smoke tests ---


class TestV2Async:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_images_generate(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{V2}/images/generations").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = await async_client.images.generate(
            prompt="cat", model="sdxl", width=512, height=512, seed=1,
        )
        assert job.request_id == "v2-job-123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_balance(self, async_client: AsyncDeapiClient) -> None:
        respx.get(f"{V2}/account/balance").mock(
            return_value=httpx.Response(200, json={"data": {"balance": 1.0}})
        )
        bal = await async_client.balance()
        assert bal.balance == 1.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_unified_transcription(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{V2}/audio/transcriptions").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = await async_client.transcription.create(
            source_url="https://x", include_ts=True, model="whisper",
        )
        assert job.request_id == "v2-job-123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_unified_prompt_enhance(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{V2}/prompts/enhancements").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        result = await async_client.prompts.enhance(
            prompt="cat", type="images.generations", model_slug="sdxl",
        )
        assert result.prompt == "enhanced"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_enhance_price(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{V2}/prompts/enhancements/price").mock(
            return_value=httpx.Response(200, json={"data": {"price": 0.002}})
        )
        price = await async_client.prompts.enhance_price(
            prompt="cat", type="images.generations", model_slug="sdxl",
        )
        assert price.price == 0.002
