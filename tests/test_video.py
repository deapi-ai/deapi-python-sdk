from __future__ import annotations

import json
import tempfile
from pathlib import Path

import httpx
import pytest
import respx

from deapi import AsyncDeapiClient, DeapiClient

TEST_BASE_URL = "https://test.deapi.ai"
TEST_API_KEY = "test-key-123"

SUBMIT_RESPONSE = {"data": {"request_id": "test-uuid-123"}}
PRICE_RESPONSE = {"data": {"price": 0.50}}


class TestVideoGenerate:
    @respx.mock
    def test_generate_returns_job(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2video").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.video.generate(
            prompt="a cat flying through space",
            model="ltx-video",
            width=512,
            height=512,
            steps=1,
            seed=42,
            frames=120,
            fps=30,
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    def test_generate_sends_correct_payload(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2video").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.video.generate(
            prompt="test prompt",
            negative_prompt="blurry",
            model="ltx-video",
            width=768,
            height=512,
            steps=1,
            seed=42,
            frames=120,
            fps=30,
            guidance=7.5,
            webhook_url="https://example.com/hook",
        )
        body = json.loads(route.calls.last.request.content)
        assert body["prompt"] == "test prompt"
        assert body["negative_prompt"] == "blurry"
        assert body["model"] == "ltx-video"
        assert body["width"] == 768
        assert body["height"] == 512
        assert body["steps"] == 1
        assert body["seed"] == 42
        assert body["frames"] == 120
        assert body["fps"] == 30
        assert body["guidance"] == 7.5
        assert body["webhook_url"] == "https://example.com/hook"

    @respx.mock
    def test_generate_minimal_payload(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2video").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.video.generate(
            prompt="test", model="ltx", width=512, height=512,
            steps=1, seed=1, frames=60, fps=24,
        )
        body = json.loads(route.calls.last.request.content)
        assert "negative_prompt" not in body
        assert "guidance" not in body
        assert "webhook_url" not in body

    @respx.mock
    def test_generate_price(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2video/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.video.generate_price(
            prompt="test", model="ltx", width=512, height=512,
            steps=1, seed=1, frames=60, fps=24,
        )
        assert price.price == 0.50


class TestVideoAnimate:
    @respx.mock
    def test_animate_returns_job(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img2video").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.video.animate(
            prompt="make it move",
            model="ltx-video",
            first_frame_image=b"fake-image-data",
            seed=42,
            width=512,
            height=512,
            frames=60,
            fps=24,
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    def test_animate_sends_multipart(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/img2video").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.video.animate(
            prompt="animate this",
            model="ltx-video",
            first_frame_image=b"fake-image",
            seed=42,
            width=512,
            height=512,
            frames=60,
            fps=24,
        )
        content_type = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in content_type

    @respx.mock
    def test_animate_with_file_path(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img2video").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake-png-data")
            tmp_path = f.name

        job = client.video.animate(
            prompt="animate",
            model="ltx-video",
            first_frame_image=tmp_path,
            seed=42,
            width=512,
            height=512,
            frames=60,
            fps=24,
        )
        assert job.request_id == "test-uuid-123"
        Path(tmp_path).unlink()

    @respx.mock
    def test_animate_with_last_frame(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img2video").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.video.animate(
            prompt="morph between frames",
            model="ltx-video",
            first_frame_image=b"first-frame",
            last_frame_image=b"last-frame",
            seed=42,
            width=512,
            height=512,
            frames=60,
            fps=24,
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    def test_animate_with_optional_params(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/img2video").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.video.animate(
            prompt="animate",
            negative_prompt="ugly",
            model="ltx-video",
            first_frame_image=b"data",
            seed=42,
            width=512,
            height=512,
            frames=60,
            fps=24,
            guidance=3.5,
            steps=10,
            webhook_url="https://example.com/hook",
        )
        assert route.calls.last is not None

    @respx.mock
    def test_animate_price(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img2video/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.video.animate_price(
            model="ltx-video", width=512, height=512, frames=60, fps=24,
        )
        assert price.price == 0.50


class TestVideoGenerateFromAudio:
    @respx.mock
    def test_generate_from_audio_returns_job(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/aud2video").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.video.generate_from_audio(
            prompt="sync to beat",
            audio=b"fake-audio-data",
            model="ltx-audio2video",
            width=512,
            height=512,
            seed=42,
            frames=97,
            fps=24,
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    def test_generate_from_audio_sends_multipart(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/aud2video").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.video.generate_from_audio(
            prompt="test", audio=b"audio", model="ltx", width=512,
            height=512, seed=42, frames=97, fps=24,
        )
        content_type = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in content_type

    @respx.mock
    def test_generate_from_audio_with_frame_images(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/aud2video").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.video.generate_from_audio(
            prompt="test",
            audio=b"audio-data",
            model="ltx",
            width=512,
            height=512,
            seed=42,
            frames=97,
            fps=24,
            first_frame_image=b"first-frame",
            last_frame_image=b"last-frame",
            guidance=7.5,
            steps=20,
            webhook_url="https://example.com/hook",
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    def test_generate_from_audio_price(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/aud2video/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.video.generate_from_audio_price(
            model="ltx", width=512, height=512, frames=97, fps=24,
        )
        assert price.price == 0.50

    @respx.mock
    def test_generate_from_audio_price_with_optional(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/aud2video/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.video.generate_from_audio_price(
            model="ltx", width=512, height=512, frames=97, fps=24,
            guidance=7.5, steps=20,
        )
        assert price.price == 0.50
        body = json.loads(route.calls.last.request.content)
        assert body["guidance"] == 7.5
        assert body["steps"] == 20


class TestVideoReplace:
    @respx.mock
    def test_replace_returns_job(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/videos/replace").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.video.replace(
            video=b"fake-video-data",
            ref_image=b"fake-ref-image",
            model="vid-replace-model",
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    def test_replace_sends_multipart(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/videos/replace").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.video.replace(
            video=b"video", ref_image=b"image", model="vid-replace",
        )
        content_type = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in content_type

    @respx.mock
    def test_replace_with_all_params(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/videos/replace").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.video.replace(
            video=b"video",
            ref_image=b"image",
            model="vid-replace",
            prompt="replace the face",
            width=512,
            height=512,
            steps=8,
            seed=42,
            webhook_url="https://example.com/hook",
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    def test_replace_price_with_video(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/videos/replace/price").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.video.replace_price(
            model="vid-replace", video=b"fake-video",
        )
        assert price.price == 0.50

    def test_replace_price_requires_video_or_duration(self, client: DeapiClient) -> None:
        with pytest.raises(ValueError, match="Either 'video'"):
            client.video.replace_price(model="vid-replace")

    @respx.mock
    def test_replace_price_with_duration(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/videos/replace/price").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.video.replace_price(
            model="vid-replace", duration=10.5, width=512, height=512,
        )
        assert price.price == 0.50


class TestVideoUpscale:
    @respx.mock
    def test_upscale_returns_job(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/vid-upscale").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.video.upscale(
            video=b"fake-video-data",
            model="realesrgan-video",
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    def test_upscale_sends_multipart(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/vid-upscale").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.video.upscale(video=b"fake-video", model="realesrgan-video")
        content_type = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in content_type

    @respx.mock
    def test_upscale_price_with_dimensions(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/vid-upscale/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.video.upscale_price(
            model="realesrgan-video", width=1920, height=1080,
        )
        assert price.price == 0.50

    @respx.mock
    def test_upscale_price_with_file(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/vid-upscale/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.video.upscale_price(
            model="realesrgan-video", video=b"fake-video",
        )
        assert price.price == 0.50


class TestVideoRemoveBackground:
    @respx.mock
    def test_rmbg_returns_job(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/vid-rmbg").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.video.remove_background(
            video=b"fake-video-data",
            model="rmbg-video",
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    def test_rmbg_sends_multipart(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/vid-rmbg").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.video.remove_background(video=b"fake-video", model="rmbg-video")
        content_type = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in content_type

    @respx.mock
    def test_rmbg_with_webhook(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/vid-rmbg").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.video.remove_background(
            video=b"fake-video",
            model="rmbg-video",
            webhook_url="https://example.com/hook",
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    def test_rmbg_price_with_dimensions(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/vid-rmbg/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.video.remove_background_price(
            model="rmbg-video", width=1920, height=1080,
        )
        assert price.price == 0.50


class TestAsyncVideo:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_generate(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2video").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = await async_client.video.generate(
            prompt="test", model="ltx", width=512, height=512,
            steps=1, seed=1, frames=60, fps=24,
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_animate(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img2video").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = await async_client.video.animate(
            prompt="animate",
            model="ltx-video",
            first_frame_image=b"fake-image",
            seed=42,
            width=512,
            height=512,
            frames=60,
            fps=24,
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_upscale(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/vid-upscale").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = await async_client.video.upscale(
            video=b"fake-video", model="realesrgan-video",
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_remove_background(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/vid-rmbg").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = await async_client.video.remove_background(
            video=b"fake-video", model="rmbg-video",
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_generate_price(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2video/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = await async_client.video.generate_price(
            prompt="test", model="ltx", width=512, height=512,
            steps=1, seed=1, frames=60, fps=24,
        )
        assert price.price == 0.50

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_animate_price(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img2video/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = await async_client.video.animate_price(
            model="ltx-video", width=512, height=512, frames=60, fps=24,
        )
        assert price.price == 0.50

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_generate_from_audio(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/aud2video").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = await async_client.video.generate_from_audio(
            prompt="sync to beat", audio=b"fake-audio",
            model="ltx", width=512, height=512,
            seed=42, frames=97, fps=24,
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_generate_from_audio_price(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/aud2video/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = await async_client.video.generate_from_audio_price(
            model="ltx", width=512, height=512, frames=97, fps=24,
        )
        assert price.price == 0.50

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_replace(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/videos/replace").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = await async_client.video.replace(
            video=b"fake-video", ref_image=b"fake-image",
            model="vid-replace",
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_replace_price(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/videos/replace/price").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = await async_client.video.replace_price(
            model="vid-replace", video=b"fake-video",
        )
        assert price.price == 0.50
