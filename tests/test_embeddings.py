from __future__ import annotations

import json

import httpx
import pytest
import respx

from deapi import AsyncDeapiClient, DeapiClient

TEST_BASE_URL = "https://test.deapi.ai"
TEST_API_KEY = "test-key-123"

SUBMIT_RESPONSE = {"data": {"request_id": "test-uuid-123"}}
PRICE_RESPONSE = {"data": {"price": 0.01}}


class TestEmbeddingsCreate:
    @respx.mock
    def test_create_with_string_input(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2embedding").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.embeddings.create(
            input="hello world",
            model="bge-m3",
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    def test_create_with_list_input(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2embedding").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.embeddings.create(
            input=["hello", "world"],
            model="bge-m3",
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    def test_create_sends_correct_payload(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2embedding").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.embeddings.create(
            input=["text1", "text2"],
            model="bge-m3",
            return_result_in_response=True,
            webhook_url="https://example.com/hook",
        )
        body = json.loads(route.calls.last.request.content)
        assert body["input"] == ["text1", "text2"]
        assert body["model"] == "bge-m3"
        assert body["return_result_in_response"] is True
        assert body["webhook_url"] == "https://example.com/hook"

    @respx.mock
    def test_create_minimal_payload(self, client: DeapiClient) -> None:
        route = respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2embedding").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        client.embeddings.create(input="test", model="bge-m3")
        body = json.loads(route.calls.last.request.content)
        assert "return_result_in_response" not in body
        assert "webhook_url" not in body

    @respx.mock
    def test_create_price_with_string(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2embedding/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.embeddings.create_price(
            input="hello world",
            model="bge-m3",
        )
        assert price.price == 0.01

    @respx.mock
    def test_create_price_with_list(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2embedding/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = client.embeddings.create_price(
            input=["hello", "world"],
            model="bge-m3",
        )
        assert price.price == 0.01


class TestAsyncEmbeddings:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_create(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2embedding").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = await async_client.embeddings.create(
            input="hello world",
            model="bge-m3",
        )
        assert job.request_id == "test-uuid-123"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_create_price(self, async_client: AsyncDeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2embedding/price-calculation").mock(
            return_value=httpx.Response(200, json=PRICE_RESPONSE)
        )
        price = await async_client.embeddings.create_price(
            input=["hello", "world"],
            model="bge-m3",
        )
        assert price.price == 0.01
