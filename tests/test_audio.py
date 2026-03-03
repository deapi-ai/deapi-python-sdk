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

SUBMIT_RESPONSE = {"data": {"request_id": "audio-job-123"}}
PRICE_RESPONSE = {"data": {"price": 0.10}}


class TestTTS:
    @respx.mock
    def test_synthesize_json(self, client: DeapiClient) -> None:
        """TTS without ref_audio should send JSON."""
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2audio").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.audio.synthesize(
            text="Hello world",
            model="tts-v1",
            lang="en",
            format="mp3",
            speed=1.0,
            sample_rate=22050,
            mode="custom_voice",
            voice="alloy",
        )
        assert job.request_id == "audio-job-123"
        request = route.calls.last.request
        content_type = request.headers.get("content-type", "")
        assert "application/json" in content_type
        body = json.loads(request.content)
        assert body["text"] == "Hello world"
        assert body["model"] == "tts-v1"
        assert body["mode"] == "custom_voice"
        assert body["voice"] == "alloy"

    @respx.mock
    def test_synthesize_voice_clone_multipart(self, client: DeapiClient) -> None:
        """TTS with ref_audio should send multipart."""
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2audio").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.audio.synthesize(
            text="Clone this voice",
            model="tts-v1",
            lang="en",
            format="wav",
            speed=1.0,
            sample_rate=22050,
            mode="voice_clone",
            ref_audio=b"fake-audio-bytes",
            ref_text="reference transcript",
        )
        assert job.request_id == "audio-job-123"
        request = route.calls.last.request
        content_type = request.headers.get("content-type", "")
        assert "multipart/form-data" in content_type

    @respx.mock
    def test_synthesize_price_with_text(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2audio/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.audio.synthesize_price(
            text="Hello world",
            model="tts-v1",
            lang="en",
            format="mp3",
            speed=1.0,
            sample_rate=22050,
        )
        assert price.price == 0.10

    @respx.mock
    def test_synthesize_price_with_count(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2audio/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.audio.synthesize_price(
            count_text=500,
            model="tts-v1",
            lang="en",
            format="mp3",
            speed=1.0,
            sample_rate=22050,
        )
        assert price.price == 0.10
        body = json.loads(route.calls.last.request.content)
        assert body["count_text"] == 500
        assert "text" not in body

    @respx.mock
    def test_synthesize_minimal(self, client: DeapiClient) -> None:
        """TTS with only required fields."""
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2audio").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.audio.synthesize(
            text="test", model="tts-v1", lang="en",
            format="mp3", speed=1.0, sample_rate=22050,
        )
        body = json.loads(route.calls.last.request.content)
        assert "mode" not in body
        assert "voice" not in body
        assert "webhook_url" not in body


class TestMusic:
    @respx.mock
    def test_compose(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2music").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.audio.compose(
            caption="upbeat electronic dance track",
            model="music-v1",
            duration=30.0,
            inference_steps=50,
            guidance_scale=7.0,
            seed=42,
            format="mp3",
            lyrics="la la la",
            bpm=120,
        )
        assert job.request_id == "audio-job-123"
        body = json.loads(route.calls.last.request.content)
        assert body["caption"] == "upbeat electronic dance track"
        assert body["duration"] == 30.0
        assert body["lyrics"] == "la la la"
        assert body["bpm"] == 120

    @respx.mock
    def test_compose_minimal(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2music").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.audio.compose(
            caption="test", model="music-v1", duration=10.0,
            inference_steps=20, guidance_scale=5.0, seed=1, format="wav",
        )
        body = json.loads(route.calls.last.request.content)
        assert "lyrics" not in body
        assert "bpm" not in body
        assert "keyscale" not in body

    @respx.mock
    def test_compose_price(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2music/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.audio.compose_price(
            model="music-v1", duration=30.0, inference_steps=50,
        )
        assert price.price == 0.10
        body = json.loads(route.calls.last.request.content)
        assert set(body.keys()) == {"model", "duration", "inference_steps"}


class TestAsyncAudio:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_synthesize(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2audio").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = await async_client.audio.synthesize(
            text="hello", model="tts-v1", lang="en",
            format="mp3", speed=1.0, sample_rate=22050,
        )
        assert job.request_id == "audio-job-123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_compose(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2music").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = await async_client.audio.compose(
            caption="test", model="music-v1", duration=10.0,
            inference_steps=20, guidance_scale=5.0, seed=1, format="wav",
        )
        assert job.request_id == "audio-job-123"
