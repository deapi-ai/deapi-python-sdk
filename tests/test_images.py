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
PRICE_RESPONSE = {"data": {"price": 0.25}}


class TestImageGeneration:
    @respx.mock
    def test_generate_returns_job(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2img").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.images.generate(
            prompt="a cat in space",
            model="sdxl",
            width=1024,
            height=1024,
            seed=42,
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    def test_generate_sends_correct_payload(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2img").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.images.generate(
            prompt="test prompt",
            negative_prompt="blurry",
            model="sdxl",
            width=1024,
            height=768,
            seed=42,
            guidance=7.5,
            steps=30,
            loras=[{"name": "detail", "weight": 0.8}],
            webhook_url="https://example.com/hook",
        )
        request = route.calls.last.request
        import json
        body = json.loads(request.content)
        assert body["prompt"] == "test prompt"
        assert body["negative_prompt"] == "blurry"
        assert body["model"] == "sdxl"
        assert body["width"] == 1024
        assert body["height"] == 768
        assert body["seed"] == 42
        assert body["guidance"] == 7.5
        assert body["steps"] == 30
        assert body["loras"] == [{"name": "detail", "weight": 0.8}]
        assert body["webhook_url"] == "https://example.com/hook"

    @respx.mock
    def test_generate_minimal_payload(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2img").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.images.generate(prompt="test", model="sdxl", width=512, height=512, seed=1)
        import json
        body = json.loads(route.calls.last.request.content)
        assert "negative_prompt" not in body
        assert "loras" not in body
        assert "guidance" not in body
        assert "steps" not in body
        assert "webhook_url" not in body

    @respx.mock
    def test_generate_price(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2img/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.images.generate_price(
            prompt="test", model="sdxl", width=1024, height=1024, seed=42,
        )
        assert price.price == 0.25


class TestImageTransform:
    @respx.mock
    def test_transform_with_file_path(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img2img").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake-png-data")
            tmp_path = f.name

        job = client.images.transform(
            prompt="make it a painting",
            image=tmp_path,
            model="sdxl-img2img",
            steps=30,
            seed=42,
        )
        assert job.request_id == "test-uuid-123"
        Path(tmp_path).unlink()

    @respx.mock
    def test_transform_with_bytes(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img2img").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.images.transform(
            prompt="enhance",
            image=b"raw-image-bytes",
            model="sdxl-img2img",
            steps=20,
            seed=1,
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    def test_transform_with_multiple_images(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img2img").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1:
            f1.write(b"image1")
            path1 = f1.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
            f2.write(b"image2")
            path2 = f2.name

        job = client.images.transform(
            prompt="blend",
            images=[path1, path2],
            model="sdxl-img2img",
            steps=20,
            seed=1,
        )
        assert job.request_id == "test-uuid-123"
        Path(path1).unlink()
        Path(path2).unlink()

    @respx.mock
    def test_transform_sends_multipart(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/img2img").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.images.transform(
            prompt="test",
            image=b"fake-data",
            model="sdxl-img2img",
            steps=20,
            seed=1,
        )
        request = route.calls.last.request
        content_type = request.headers.get("content-type", "")
        assert "multipart/form-data" in content_type

    @respx.mock
    def test_transform_price(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img2img/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.images.transform_price(
            prompt="test", model="sdxl-img2img", steps=30, seed=42,
        )
        assert price.price == 0.25


class TestImageUpscale:
    @respx.mock
    def test_upscale_returns_job(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img-upscale").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.images.upscale(image=b"fake-image", model="realesrgan-x4")
        assert job.request_id == "test-uuid-123"

    @respx.mock
    def test_upscale_sends_multipart(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/img-upscale").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.images.upscale(image=b"fake-image", model="realesrgan-x4")
        content_type = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in content_type

    @respx.mock
    def test_upscale_with_file_path(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img-upscale").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"fake-png-data")
            tmp_path = f.name
        job = client.images.upscale(image=tmp_path, model="realesrgan-x4")
        assert job.request_id == "test-uuid-123"
        Path(tmp_path).unlink()

    @respx.mock
    def test_upscale_with_webhook(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img-upscale").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.images.upscale(
            image=b"fake-image", model="realesrgan-x4",
            webhook_url="https://example.com/hook",
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    def test_upscale_price_with_dimensions(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img-upscale/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.images.upscale_price(
            model="realesrgan-x4", width=1024, height=1024,
        )
        assert price.price == 0.25

    @respx.mock
    def test_upscale_price_with_file(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img-upscale/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.images.upscale_price(
            model="realesrgan-x4", image=b"fake-image",
        )
        assert price.price == 0.25


class TestImageRemoveBackground:
    @respx.mock
    def test_rmbg_returns_job(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img-rmbg").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.images.remove_background(image=b"fake-image", model="rmbg-1.4")
        assert job.request_id == "test-uuid-123"

    @respx.mock
    def test_rmbg_sends_multipart(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/img-rmbg").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.images.remove_background(image=b"fake-image", model="rmbg-1.4")
        content_type = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in content_type

    @respx.mock
    def test_rmbg_price_with_dimensions(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img-rmbg/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.images.remove_background_price(
            model="rmbg-1.4", width=512, height=512,
        )
        assert price.price == 0.25


class TestAsyncImages:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_generate(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2img").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = await async_client.images.generate(
            prompt="test", model="sdxl", width=512, height=512, seed=1,
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_generate_price(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2img/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = await async_client.images.generate_price(
            prompt="test", model="sdxl", width=512, height=512, seed=1,
        )
        assert price.price == 0.25

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_upscale(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img-upscale").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = await async_client.images.upscale(image=b"fake-image", model="realesrgan-x4")
        assert job.request_id == "test-uuid-123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_remove_background(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img-rmbg").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = await async_client.images.remove_background(image=b"fake-image", model="rmbg-1.4")
        assert job.request_id == "test-uuid-123"
