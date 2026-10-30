from __future__ import annotations

import json

import httpx
import pytest
import respx

from deapi_sdk import AsyncDeapiClient, DeapiClient

TEST_BASE_URL = "https://test.deapi.ai"
TEST_API_KEY = "test-key-123"

SUBMIT_RESPONSE = {"data": {"request_id": "transcription-job-123"}}
PRICE_RESPONSE = {"data": {"price": 0.05}}


class TestUnifiedTranscribe:
    @respx.mock
    def test_create_with_url(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v2/audio/transcriptions").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.transcription.create(
            source_url="https://youtube.com/watch?v=abc123",
            include_ts=True,
            model="whisper-v3",
            return_result_in_response=True,
        )
        assert job.request_id == "transcription-job-123"
        body = json.loads(route.calls.last.request.content)
        assert body["source_url"] == "https://youtube.com/watch?v=abc123"
        assert body["include_ts"] is True
        assert body["return_result_in_response"] is True

    @respx.mock
    def test_create_with_file(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v2/audio/transcriptions").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.transcription.create(
            source_file=b"fake-audio-data",
            include_ts=False,
            model="whisper-v3",
        )
        assert job.request_id == "transcription-job-123"
        content_type = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in content_type

    def test_create_both_raises(self, client: DeapiClient) -> None:
        with pytest.raises(ValueError, match="not both"):
            client.transcription.create(
                source_url="https://example.com",
                source_file=b"data",
                include_ts=True,
                model="whisper-v3",
            )

    def test_create_neither_raises(self, client: DeapiClient) -> None:
        with pytest.raises(ValueError, match="is required"):
            client.transcription.create(
                include_ts=True,
                model="whisper-v3",
            )

    @respx.mock
    def test_create_price_with_url(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v2/audio/transcriptions/price").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.transcription.create_price(
            source_url="https://youtube.com/watch?v=abc",
            include_ts=True,
            model="whisper-v3",
        )
        assert price.price == 0.05

    @respx.mock
    def test_create_price_with_duration(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v2/audio/transcriptions/price").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.transcription.create_price(
            duration_seconds=120.0,
            include_ts=True,
            model="whisper-v3",
        )
        assert price.price == 0.05
        body = json.loads(route.calls.last.request.content)
        assert body["duration_seconds"] == 120.0


class TestAsyncTranscription:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_create(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v2/audio/transcriptions").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = await async_client.transcription.create(
            source_url="https://youtube.com/watch?v=abc",
            include_ts=True,
            model="whisper-v3",
        )
        assert job.request_id == "transcription-job-123"
