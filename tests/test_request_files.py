"""Low-level ``request()`` / ``post()`` accept every multipart ``files`` shape httpx does.

This module is also type-checked with mypy --strict in CI, so the calls below double as a typing test
for ``deapi_sdk._client.RequestFiles``.
"""

from __future__ import annotations

import io

import httpx
import respx

from deapi_sdk import AsyncDeapiClient, DeapiClient
from deapi_sdk._client import RequestFiles
from tests.conftest import TEST_API_KEY, TEST_BASE_URL

URL = f"{TEST_BASE_URL}/api/v2/images/upscales"
OK = httpx.Response(200, json={"data": {"request_id": "req-1"}})


def _parts() -> list[RequestFiles]:
    return [
        {"image": ("a.png", io.BytesIO(b"PNGDATA-1"), "image/png")},
        [("image", ("a.png", io.BytesIO(b"PNGDATA-1"), "image/png"))],
        [("image", ("a.png", b"PNGDATA-1", "image/png"))],
        {"image": io.BytesIO(b"PNGDATA-1")},
    ]


class TestRequestFilesShapes:
    @respx.mock
    def test_sync_post_accepts_mapping_and_sequence_with_file_objects(self) -> None:
        route = respx.post(URL).mock(return_value=OK)
        with DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, fill_model_defaults=False) as client:
            for files in _parts():
                client._http_client.post(URL, data={"model": "m"}, files=files)
                assert b"PNGDATA-1" in route.calls.last.request.content
        assert route.call_count == 4

    @respx.mock
    async def test_async_post_accepts_mapping_and_sequence_with_file_objects(self) -> None:
        route = respx.post(URL).mock(return_value=OK)
        async with AsyncDeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, fill_model_defaults=False) as client:
            for files in _parts():
                await client._http_client.post(URL, data={"model": "m"}, files=files)
                assert b"PNGDATA-1" in route.calls.last.request.content
        assert route.call_count == 4

    @respx.mock
    def test_rate_limited_resend_sends_the_whole_file_again(self) -> None:
        route = respx.post(URL).mock(
            side_effect=[httpx.Response(429, headers={"Retry-After": "1"}, json={"message": "slow down"}), OK]
        )
        with DeapiClient(
            api_key=TEST_API_KEY, base_url=TEST_BASE_URL, fill_model_defaults=False, max_retry_after=0.01
        ) as client:
            files: RequestFiles = {"image": ("a.png", io.BytesIO(b"PNGDATA-RETRY"), "image/png")}
            client._http_client.post(URL, data={"model": "m"}, files=files)
        assert route.call_count == 2
        assert all(b"PNGDATA-RETRY" in call.request.content for call in route.calls)
