"""Tests for v2 API support.

Verifies that:
* v2 endpoint paths get hit (renames mapped correctly).
* Unified endpoints work (transcription, prompt enhancer).
* Legacy methods emit DeprecationWarning and route to unified endpoints.
* The polling status URL uses ``/api/v2/jobs/{id}``.
* The default api_version is now ``v2``.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from deapi import AsyncDeapiClient, DeapiClient

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

    def test_v1_prefix_unchanged(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, api_version="v1")
        assert client._config.api_prefix == "/api/v1/client"
        client.close()


# --- Path renames (v2 routes get hit) ---


class TestV2Paths:
    @respx.mock
    def test_images_generate_hits_v2_path(self, client_v2: DeapiClient) -> None:
        respx.post(f"{V2}/images/generations").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client_v2.images.generate(
            prompt="cat in space", model="sdxl", width=512, height=512, seed=1,
        )
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_images_edit_hits_v2_path(self, client_v2: DeapiClient) -> None:
        respx.post(f"{V2}/images/edits").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client_v2.images.transform(
            prompt="paint blue", model="sdxl-edit", image=b"fake-image",
            steps=4, seed=1,
        )
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_images_upscale_hits_v2_path(self, client_v2: DeapiClient) -> None:
        respx.post(f"{V2}/images/upscales").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client_v2.images.upscale(image=b"fake-image", model="real-esrgan")
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_images_rmbg_hits_v2_path(self, client_v2: DeapiClient) -> None:
        respx.post(f"{V2}/images/background-removals").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client_v2.images.remove_background(image=b"fake-image", model="rembg")
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_video_generate_hits_v2_path(self, client_v2: DeapiClient) -> None:
        respx.post(f"{V2}/videos/generations").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client_v2.video.generate(
            prompt="ocean", model="wan2.2", width=512, height=512,
            steps=4, seed=1, frames=24, fps=24,
        )
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_video_animate_hits_v2_path(self, client_v2: DeapiClient) -> None:
        respx.post(f"{V2}/videos/animations").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client_v2.video.animate(
            prompt="zoom in", model="wan-i2v", first_frame_image=b"img",
            seed=1, width=512, height=512, frames=24, fps=24,
        )
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_video_upscale_hits_v2_path(self, client_v2: DeapiClient) -> None:
        respx.post(f"{V2}/videos/upscales").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client_v2.video.upscale(video=b"vid", model="real-esrgan")
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_video_rmbg_hits_v2_path(self, client_v2: DeapiClient) -> None:
        respx.post(f"{V2}/videos/background-removals").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client_v2.video.remove_background(video=b"vid", model="rembg")
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_video_replace_hits_v2_path(self, client_v2: DeapiClient) -> None:
        respx.post(f"{V2}/videos/replacements").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client_v2.video.replace(video=b"vid", ref_image=b"ref", model="wan-replace")
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_video_audio_sync_hits_v2_path(self, client_v2: DeapiClient) -> None:
        respx.post(f"{V2}/videos/audio-syncs").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client_v2.video.generate_from_audio(
            prompt="dance", audio=b"audio", model="wan-a2v",
            width=512, height=512, seed=1, frames=24, fps=24,
        )
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_audio_speech_hits_v2_path(self, client_v2: DeapiClient) -> None:
        respx.post(f"{V2}/audio/speech").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client_v2.audio.synthesize(
            text="hello world", model="kokoro", lang="en", format="wav",
            speed=1.0, sample_rate=24000.0, voice="alpha",
        )
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_audio_music_hits_v2_path(self, client_v2: DeapiClient) -> None:
        respx.post(f"{V2}/audio/music").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client_v2.audio.compose(
            caption="jazz beat", model="musicgen", duration=30.0,
            inference_steps=50, guidance_scale=3.0, seed=1, format="wav",
        )
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_embeddings_hits_v2_path(self, client_v2: DeapiClient) -> None:
        respx.post(f"{V2}/embeddings").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client_v2.embeddings.create(input="hello", model="bge-m3")
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_ocr_hits_v2_path(self, client_v2: DeapiClient) -> None:
        respx.post(f"{V2}/images/ocr").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client_v2.ocr.extract(image=b"img", model="paddleocr")
        assert job.request_id == "v2-job-123"

    @respx.mock
    def test_balance_hits_v2_path(self, client_v2: DeapiClient) -> None:
        respx.get(f"{V2}/account/balance").mock(
            return_value=httpx.Response(200, json={"data": {"balance": 42.0}})
        )
        bal = client_v2.balance()
        assert bal.balance == 42.0

    @respx.mock
    def test_models_hits_v2_path(self, client_v2: DeapiClient) -> None:
        respx.get(f"{V2}/models").mock(
            return_value=httpx.Response(200, json={
                "data": [], "links": {}, "meta": {"current_page": 1, "last_page": 1, "per_page": 15, "total": 0},
            })
        )
        result = client_v2.models.list()
        assert result.meta.total == 0


# --- Job polling URL ---


class TestV2Polling:
    @respx.mock
    def test_polling_uses_jobs_path(self, client_v2: DeapiClient) -> None:
        respx.post(f"{V2}/images/generations").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        respx.get(f"{V2}/jobs/v2-job-123").mock(
            return_value=httpx.Response(200, json={
                "data": {"status": "done", "result_url": "https://cdn/image.png", "progress": 1.0},
            })
        )
        job = client_v2.images.generate(
            prompt="cat", model="sdxl", width=512, height=512, seed=1,
        )
        result = job.status()
        assert result.result_url == "https://cdn/image.png"


# --- Unified transcription ---


class TestV2Transcription:
    @respx.mock
    def test_create_with_url(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/audio/transcriptions").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = client_v2.transcription.create(
            source_url="https://example.com/a.mp4", include_ts=True, model="whisper",
        )
        assert job.request_id == "v2-job-123"
        body = json.loads(route.calls.last.request.content)
        assert body["source_url"] == "https://example.com/a.mp4"

    @respx.mock
    def test_create_with_file_uses_multipart(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/audio/transcriptions").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        client_v2.transcription.create(
            source_file=b"audio-bytes", include_ts=False, model="whisper",
        )
        ct = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in ct

    @respx.mock
    def test_create_price(self, client_v2: DeapiClient) -> None:
        respx.post(f"{V2}/audio/transcriptions/price").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client_v2.transcription.create_price(
            include_ts=True, model="whisper", duration_seconds=60.0,
        )
        assert price.price == 0.05

    @respx.mock
    def test_legacy_from_video_url_warns_and_routes(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/audio/transcriptions").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        with pytest.warns(DeprecationWarning, match="unified transcription"):
            job = client_v2.transcription.from_video_url(
                video_url="https://example.com/v.mp4", include_ts=True, model="whisper",
            )
        assert job.request_id == "v2-job-123"
        body = json.loads(route.calls.last.request.content)
        assert body["source_url"] == "https://example.com/v.mp4"

    @respx.mock
    def test_legacy_from_audio_file_warns_and_routes(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/audio/transcriptions").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        with pytest.warns(DeprecationWarning):
            client_v2.transcription.from_audio_file(
                audio=b"audio", include_ts=True, model="whisper",
            )
        ct = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in ct

    @respx.mock
    def test_legacy_from_audio_url_price_warns_and_routes(self, client_v2: DeapiClient) -> None:
        respx.post(f"{V2}/audio/transcriptions/price").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        with pytest.warns(DeprecationWarning):
            price = client_v2.transcription.from_audio_url_price(
                include_ts=True, model="whisper", audio_url="https://x", duration_seconds=10.0,
            )
        assert price.price == 0.05


# --- Unified prompt enhancer ---


class TestV2Prompts:
    @respx.mock
    def test_enhance_unified(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/prompts/enhancements").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        result = client_v2.prompts.enhance(
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
    def test_enhance_with_image(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/prompts/enhancements").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        client_v2.prompts.enhance(
            prompt="a cat", type="images.edits", model_slug="sdxl-edit", image=b"img",
        )
        # Multipart should contain the image as a file part
        body = route.calls.last.request.content
        assert b"image" in body

    @respx.mock
    def test_enhance_price(self, client_v2: DeapiClient) -> None:
        respx.post(f"{V2}/prompts/enhancements/price").mock(
            return_value=httpx.Response(200, json={"data": {"price": 0.001}})
        )
        price = client_v2.prompts.enhance_price(
            prompt="a cat", type="images.generations", model_slug="sdxl",
        )
        assert price.price == 0.001

    @respx.mock
    def test_legacy_enhance_image_warns_and_routes(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/prompts/enhancements").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        with pytest.warns(DeprecationWarning, match="unified prompt enhancer"):
            client_v2.prompts.enhance_image(prompt="cat", model_slug="sdxl")
        body = route.calls.last.request.content
        assert b"images.generations" in body

    @respx.mock
    def test_legacy_enhance_video_with_image_routes_animations(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/prompts/enhancements").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        with pytest.warns(DeprecationWarning):
            client_v2.prompts.enhance_video(prompt="zoom", model_slug="wan-i2v", image=b"img")
        body = route.calls.last.request.content
        assert b"videos.animations" in body

    @respx.mock
    def test_legacy_enhance_video_no_image_routes_generations(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/prompts/enhancements").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        with pytest.warns(DeprecationWarning):
            client_v2.prompts.enhance_video(prompt="ocean", model_slug="wan-t2v")
        body = route.calls.last.request.content
        assert b"videos.generations" in body

    @respx.mock
    def test_legacy_enhance_speech_warns(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/prompts/enhancements").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        with pytest.warns(DeprecationWarning):
            client_v2.prompts.enhance_speech(prompt="hello", model_slug="kokoro")
        body = route.calls.last.request.content
        assert b"audio.speech" in body

    @respx.mock
    def test_legacy_enhance_image2image_warns(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/prompts/enhancements").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        with pytest.warns(DeprecationWarning):
            client_v2.prompts.enhance_image2image(
                prompt="paint", image=b"img", model_slug="sdxl-edit",
            )
        body = route.calls.last.request.content
        assert b"images.edits" in body

    def test_samples_raises_not_implemented(self, client_v2: DeapiClient) -> None:
        with pytest.raises(NotImplementedError, match="removed in v2"):
            client_v2.prompts.samples(type="text2image")

    def test_samples_price_raises_not_implemented(self, client_v2: DeapiClient) -> None:
        with pytest.raises(NotImplementedError, match="removed in v2"):
            client_v2.prompts.samples_price(type="text2image")


# --- Async parity smoke tests ---


class TestV2Async:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_images_generate(self, async_client_v2: AsyncDeapiClient) -> None:
        respx.post(f"{V2}/images/generations").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = await async_client_v2.images.generate(
            prompt="cat", model="sdxl", width=512, height=512, seed=1,
        )
        assert job.request_id == "v2-job-123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_balance(self, async_client_v2: AsyncDeapiClient) -> None:
        respx.get(f"{V2}/account/balance").mock(
            return_value=httpx.Response(200, json={"data": {"balance": 1.0}})
        )
        bal = await async_client_v2.balance()
        assert bal.balance == 1.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_unified_transcription(self, async_client_v2: AsyncDeapiClient) -> None:
        respx.post(f"{V2}/audio/transcriptions").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        job = await async_client_v2.transcription.create(
            source_url="https://x", include_ts=True, model="whisper",
        )
        assert job.request_id == "v2-job-123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_unified_prompt_enhance(self, async_client_v2: AsyncDeapiClient) -> None:
        respx.post(f"{V2}/prompts/enhancements").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        result = await async_client_v2.prompts.enhance(
            prompt="cat", type="images.generations", model_slug="sdxl",
        )
        assert result.prompt == "enhanced"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_legacy_transcription_warns_and_routes(self, async_client_v2: AsyncDeapiClient) -> None:
        route = respx.post(f"{V2}/audio/transcriptions").mock(
            return_value=httpx.Response(200, json=JOB_RESPONSE)
        )
        with pytest.warns(DeprecationWarning):
            await async_client_v2.transcription.from_audio_url(
                audio_url="https://x", include_ts=True, model="whisper",
            )
        body = json.loads(route.calls.last.request.content)
        assert body["source_url"] == "https://x"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_enhance_price(self, async_client_v2: AsyncDeapiClient) -> None:
        respx.post(f"{V2}/prompts/enhancements/price").mock(
            return_value=httpx.Response(200, json={"data": {"price": 0.002}})
        )
        price = await async_client_v2.prompts.enhance_price(
            prompt="cat", type="images.generations", model_slug="sdxl",
        )
        assert price.price == 0.002

    @pytest.mark.asyncio
    async def test_async_samples_raises_not_implemented(self, async_client_v2: AsyncDeapiClient) -> None:
        with pytest.raises(NotImplementedError, match="removed in v2"):
            await async_client_v2.prompts.samples(type="text2image")
