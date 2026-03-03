from __future__ import annotations

import httpx
import pytest
import respx

from deapi import AsyncDeapiClient, DeapiClient

TEST_BASE_URL = "https://test.deapi.ai"

SUBMIT_RESPONSE = {"data": {"request_id": "ocr-uuid-123"}}
PRICE_RESPONSE = {"data": {"price": 0.003}}


class TestOCRExtract:
    @respx.mock
    def test_extract_returns_job(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img2txt").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.ocr.extract(image=b"fake-png-data", model="Nanonets_Ocr_S_F16")
        assert job.request_id == "ocr-uuid-123"

    @respx.mock
    def test_extract_sends_multipart(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/img2txt").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.ocr.extract(image=b"fake-png-data", model="Nanonets_Ocr_S_F16")
        content_type = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in content_type

    @respx.mock
    def test_extract_with_all_params(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/img2txt").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.ocr.extract(
            image=b"fake-png-data",
            model="Nanonets_Ocr_S_F16",
            language="en",
            format="json",
            return_result_in_response=True,
            webhook_url="https://example.com/hook",
        )
        request = route.calls.last.request
        body = request.content.decode("utf-8", errors="replace")
        assert "Nanonets_Ocr_S_F16" in body
        assert "en" in body
        assert "json" in body

    @respx.mock
    def test_extract_minimal_params(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/img2txt").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.ocr.extract(image=b"fake-png-data", model="Nanonets_Ocr_S_F16")
        request = route.calls.last.request
        body = request.content.decode("utf-8", errors="replace")
        assert "language" not in body or body.count("language") == 0
        assert "Nanonets_Ocr_S_F16" in body


class TestOCRExtractPrice:
    @respx.mock
    def test_extract_price_with_image(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img2txt/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.ocr.extract_price(model="Nanonets_Ocr_S_F16", image=b"fake-png-data")
        assert price.price == 0.003

    @respx.mock
    def test_extract_price_with_dimensions(self, client: DeapiClient) -> None:
        import json

        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/img2txt/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.ocr.extract_price(model="Nanonets_Ocr_S_F16", width=1024, height=768)
        assert price.price == 0.003
        body = json.loads(route.calls.last.request.content)
        assert body["model"] == "Nanonets_Ocr_S_F16"
        assert body["width"] == 1024
        assert body["height"] == 768

    @respx.mock
    def test_extract_price_with_image_sends_multipart(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/img2txt/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        client.ocr.extract_price(model="Nanonets_Ocr_S_F16", image=b"fake")
        content_type = route.calls.last.request.headers.get("content-type", "")
        assert "multipart/form-data" in content_type

    @respx.mock
    def test_extract_price_without_image_sends_json(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/img2txt/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        client.ocr.extract_price(model="Nanonets_Ocr_S_F16", width=512, height=512)
        content_type = route.calls.last.request.headers.get("content-type", "")
        assert "application/json" in content_type


class TestAsyncOCR:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_extract(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img2txt").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = await async_client.ocr.extract(image=b"data", model="Nanonets_Ocr_S_F16")
        assert job.request_id == "ocr-uuid-123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_extract_price(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/img2txt/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = await async_client.ocr.extract_price(model="Nanonets_Ocr_S_F16", width=512, height=512)
        assert price.price == 0.003
