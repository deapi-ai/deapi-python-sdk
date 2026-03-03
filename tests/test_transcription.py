from __future__ import annotations

import json

import httpx
import pytest
import respx

from deapi import AsyncDeapiClient, DeapiClient

TEST_BASE_URL = "https://test.deapi.ai"
TEST_API_KEY = "test-key-123"

SUBMIT_RESPONSE = {"data": {"request_id": "transcription-job-123"}}
PRICE_RESPONSE = {"data": {"price": 0.05}}


class TestUnifiedTranscribe:
    @respx.mock
    def test_create_with_url(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/transcribe").mock(
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
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/transcribe").mock(
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
        respx.post(f"{TEST_BASE_URL}/api/v1/client/transcribe/price-calculation").mock(
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
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/transcribe/price-calculation").mock(
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


class TestVideoUrlTranscription:
    @respx.mock
    def test_from_video_url(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/vid2txt").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.transcription.from_video_url(
            video_url="https://youtube.com/watch?v=xyz",
            include_ts=True,
            model="whisper-v3",
            return_result_in_response=True,
            webhook_url="https://hooks.example.com/done",
        )
        assert job.request_id == "transcription-job-123"
        body = json.loads(route.calls.last.request.content)
        assert body["video_url"] == "https://youtube.com/watch?v=xyz"
        assert body["webhook_url"] == "https://hooks.example.com/done"

    @respx.mock
    def test_from_video_url_price_with_url(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/vid2txt/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.transcription.from_video_url_price(
            video_url="https://youtube.com/watch?v=abc",
            include_ts=True,
            model="whisper-v3",
        )
        assert price.price == 0.05

    @respx.mock
    def test_from_video_url_price_with_duration(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/vid2txt/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.transcription.from_video_url_price(
            duration_seconds=300.0,
            include_ts=False,
            model="whisper-v3",
        )
        assert price.price == 0.05
        body = json.loads(route.calls.last.request.content)
        assert body["duration_seconds"] == 300.0
        assert "video_url" not in body


class TestAudioUrlTranscription:
    @respx.mock
    def test_from_audio_url(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/aud2txt").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.transcription.from_audio_url(
            audio_url="https://twitter.com/i/spaces/abc123",
            include_ts=True,
            model="whisper-v3",
        )
        assert job.request_id == "transcription-job-123"
        body = json.loads(route.calls.last.request.content)
        assert body["audio_url"] == "https://twitter.com/i/spaces/abc123"

    @respx.mock
    def test_from_audio_url_price(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/aud2txt/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.transcription.from_audio_url_price(
            duration_seconds=600.0,
            include_ts=True,
            model="whisper-v3",
        )
        assert price.price == 0.05


class TestAudioFileTranscription:
    @respx.mock
    def test_from_audio_file(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/audiofile2txt").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.transcription.from_audio_file(
            audio=b"fake-audio-bytes",
            include_ts=True,
            model="whisper-v3",
        )
        assert job.request_id == "transcription-job-123"
        content_type = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in content_type

    @respx.mock
    def test_from_audio_file_price_with_file(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/audiofile2txt/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.transcription.from_audio_file_price(
            audio=b"fake-audio",
            include_ts=True,
            model="whisper-v3",
        )
        assert price.price == 0.05
        content_type = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in content_type

    @respx.mock
    def test_from_audio_file_price_with_duration(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/audiofile2txt/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.transcription.from_audio_file_price(
            duration_seconds=60.0,
            include_ts=True,
            model="whisper-v3",
        )
        assert price.price == 0.05
        body = json.loads(route.calls.last.request.content)
        assert body["duration_seconds"] == 60.0


class TestVideoFileTranscription:
    @respx.mock
    def test_from_video_file(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/videofile2txt").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.transcription.from_video_file(
            video=b"fake-video-bytes",
            include_ts=False,
            model="whisper-v3",
            return_result_in_response=True,
        )
        assert job.request_id == "transcription-job-123"
        content_type = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in content_type

    @respx.mock
    def test_from_video_file_price_with_duration(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/videofile2txt/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.transcription.from_video_file_price(
            duration_seconds=180.0,
            include_ts=True,
            model="whisper-v3",
        )
        assert price.price == 0.05


class TestAsyncTranscription:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_create(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/transcribe").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = await async_client.transcription.create(
            source_url="https://youtube.com/watch?v=abc",
            include_ts=True,
            model="whisper-v3",
        )
        assert job.request_id == "transcription-job-123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_from_video_url(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/vid2txt").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = await async_client.transcription.from_video_url(
            video_url="https://youtube.com/watch?v=xyz",
            include_ts=True,
            model="whisper-v3",
        )
        assert job.request_id == "transcription-job-123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_from_audio_file(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/audiofile2txt").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = await async_client.transcription.from_audio_file(
            audio=b"fake-audio",
            include_ts=True,
            model="whisper-v3",
        )
        assert job.request_id == "transcription-job-123"
