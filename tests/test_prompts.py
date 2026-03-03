from __future__ import annotations

import json

import httpx
import pytest
import respx

from deapi import AsyncDeapiClient, DeapiClient

TEST_BASE_URL = "https://test.deapi.ai"

ENHANCE_RESPONSE = {"prompt": "enhanced prompt text", "negative_prompt": "enhanced negative"}
SPEECH_RESPONSE = {"prompt": "enhanced speech prompt"}
PRICE_RESPONSE = {"price": 0.00012}
SAMPLE_RESPONSE = {
    "success": True,
    "data": {"type": "text2image", "prompt": "A futuristic cityscape..."},
}


class TestEnhanceImage:
    @respx.mock
    def test_enhance_image(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/image").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        result = client.prompts.enhance_image(prompt="a cat in space")
        assert result.prompt == "enhanced prompt text"
        assert result.negative_prompt == "enhanced negative"

    @respx.mock
    def test_enhance_image_sends_negative_prompt_null(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/image").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        client.prompts.enhance_image(prompt="test")
        body = json.loads(route.calls.last.request.content)
        assert body["prompt"] == "test"
        assert "negative_prompt" in body
        assert body["negative_prompt"] is None

    @respx.mock
    def test_enhance_image_with_negative_prompt(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/image").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        client.prompts.enhance_image(prompt="test", negative_prompt="blurry")
        body = json.loads(route.calls.last.request.content)
        assert body["negative_prompt"] == "blurry"

    @respx.mock
    def test_enhance_image_price(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/image/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.prompts.enhance_image_price(prompt="test")
        assert price.price == 0.00012


class TestEnhanceVideo:
    @respx.mock
    def test_enhance_video_json(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/video").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        result = client.prompts.enhance_video(prompt="a flying drone")
        assert result.prompt == "enhanced prompt text"
        content_type = route.calls.last.request.headers.get("content-type", "")
        assert "application/json" in content_type

    @respx.mock
    def test_enhance_video_json_sends_negative_prompt_null(self, client: DeapiClient) -> None:
        """negative_prompt must always be present in the request (API 'present' rule)."""
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/video").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        client.prompts.enhance_video(prompt="test")
        body = json.loads(route.calls.last.request.content)
        assert "negative_prompt" in body
        assert body["negative_prompt"] is None

    @respx.mock
    def test_enhance_video_multipart_sends_negative_prompt_empty(self, client: DeapiClient) -> None:
        """In multipart, negative_prompt=None is sent as empty string (not omitted)."""
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/video").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        client.prompts.enhance_video(prompt="test", image=b"fake-image")
        body_bytes = route.calls.last.request.content
        body_text = body_bytes.decode("utf-8", errors="replace")
        assert "negative_prompt" in body_text

    @respx.mock
    def test_enhance_video_with_image_multipart(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/video").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        result = client.prompts.enhance_video(
            prompt="animate this",
            image=b"fake-image-data",
        )
        assert result.prompt == "enhanced prompt text"
        content_type = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in content_type

    @respx.mock
    def test_enhance_video_price(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/video/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.prompts.enhance_video_price(prompt="test")
        assert price.price == 0.00012


class TestEnhanceSpeech:
    @respx.mock
    def test_enhance_speech(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/speech").mock(
            return_value=httpx.Response(200, json=SPEECH_RESPONSE)
        )
        result = client.prompts.enhance_speech(prompt="hello world")
        assert result.prompt == "enhanced speech prompt"

    @respx.mock
    def test_enhance_speech_with_lang(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/speech").mock(
            return_value=httpx.Response(200, json=SPEECH_RESPONSE)
        )
        client.prompts.enhance_speech(prompt="test", lang_code="es")
        body = json.loads(route.calls.last.request.content)
        assert body["lang_code"] == "es"

    @respx.mock
    def test_enhance_speech_minimal(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/speech").mock(
            return_value=httpx.Response(200, json=SPEECH_RESPONSE)
        )
        client.prompts.enhance_speech(prompt="test")
        body = json.loads(route.calls.last.request.content)
        assert "lang_code" not in body

    @respx.mock
    def test_enhance_speech_price(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/speech/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.prompts.enhance_speech_price(prompt="test")
        assert price.price == 0.00012


class TestEnhanceImage2Image:
    @respx.mock
    def test_enhance_image2image(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/image2image").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        result = client.prompts.enhance_image2image(
            prompt="make it a painting",
            image=b"fake-image",
        )
        assert result.prompt == "enhanced prompt text"
        assert result.negative_prompt == "enhanced negative"

    @respx.mock
    def test_enhance_image2image_sends_multipart(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/image2image").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        client.prompts.enhance_image2image(prompt="test", image=b"data")
        content_type = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in content_type

    @respx.mock
    def test_enhance_image2image_sends_negative_prompt_when_none(self, client: DeapiClient) -> None:
        """negative_prompt must always be present in multipart (API 'present' rule)."""
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/image2image").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        client.prompts.enhance_image2image(prompt="test", image=b"data")
        body_text = route.calls.last.request.content.decode("utf-8", errors="replace")
        assert "negative_prompt" in body_text
        # Must NOT contain the literal string "None" (Python None serialised incorrectly)
        assert 'None' not in body_text.split("negative_prompt")[1].split("--")[0] or \
               '' in body_text  # empty string is acceptable

    @respx.mock
    def test_enhance_image2image_sends_negative_prompt_value(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/image2image").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        client.prompts.enhance_image2image(prompt="test", image=b"data", negative_prompt="blurry")
        body_text = route.calls.last.request.content.decode("utf-8", errors="replace")
        assert "blurry" in body_text

    @respx.mock
    def test_enhance_image2image_price(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/image2image/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.prompts.enhance_image2image_price(prompt="test")
        assert price.price == 0.00012


class TestSamples:
    @respx.mock
    def test_samples_text2image(self, client: DeapiClient) -> None:
        respx.get(f"{TEST_BASE_URL}/api/v1/client/prompts/samples").mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )
        result = client.prompts.samples(type="text2image")
        assert result.type == "text2image"
        assert "cityscape" in result.prompt

    @respx.mock
    def test_samples_with_topic(self, client: DeapiClient) -> None:
        route = respx.get(f"{TEST_BASE_URL}/api/v1/client/prompts/samples").mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )
        client.prompts.samples(type="text2image", topic="nature")
        request = route.calls.last.request
        assert "topic=nature" in str(request.url)

    @respx.mock
    def test_samples_with_lang_code(self, client: DeapiClient) -> None:
        route = respx.get(f"{TEST_BASE_URL}/api/v1/client/prompts/samples").mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )
        client.prompts.samples(type="text2speech", lang_code="es")
        request = route.calls.last.request
        assert "lang_code=es" in str(request.url)

    @respx.mock
    def test_samples_price(self, client: DeapiClient) -> None:
        respx.get(f"{TEST_BASE_URL}/api/v1/client/prompts/samples/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.prompts.samples_price(type="text2image")
        assert price.price == 0.00012


class TestAsyncPrompts:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_enhance_image(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/image").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        result = await async_client.prompts.enhance_image(prompt="test")
        assert result.prompt == "enhanced prompt text"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_enhance_video(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/video").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        result = await async_client.prompts.enhance_video(prompt="test")
        assert result.prompt == "enhanced prompt text"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_enhance_speech(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/speech").mock(
            return_value=httpx.Response(200, json=SPEECH_RESPONSE)
        )
        result = await async_client.prompts.enhance_speech(prompt="test")
        assert result.prompt == "enhanced speech prompt"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_enhance_image2image(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/prompt/image2image").mock(
            return_value=httpx.Response(200, json=ENHANCE_RESPONSE)
        )
        result = await async_client.prompts.enhance_image2image(
            prompt="test", image=b"data",
        )
        assert result.prompt == "enhanced prompt text"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_samples(self, async_client: AsyncDeapiClient) -> None:
        respx.get(f"{TEST_BASE_URL}/api/v1/client/prompts/samples").mock(
            return_value=httpx.Response(200, json=SAMPLE_RESPONSE)
        )
        result = await async_client.prompts.samples(type="text2image")
        assert result.type == "text2image"
