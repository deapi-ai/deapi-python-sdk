from __future__ import annotations

import os
from unittest import mock

import httpx
import pytest
import respx

from deapi import (
    AsyncDeapiClient,
    DeapiClient,
)
from deapi._exceptions import (
    AccountSuspendedError,
    AuthenticationError,
    DeapiError,
    InsufficientBalanceError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)

TEST_BASE_URL = "https://test.deapi.ai"
TEST_API_KEY = "test-key-123"


class TestClientInit:
    def test_init_with_explicit_key(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL)
        assert client._config.api_key == TEST_API_KEY
        assert client._config.base_url == TEST_BASE_URL

    def test_init_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"DEAPI_API_KEY": "env-key"}):
            client = DeapiClient(base_url=TEST_BASE_URL)
            assert client._config.api_key == "env-key"

    def test_init_missing_key_raises(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DEAPI_API_KEY", None)
            with pytest.raises(ValueError, match="API key is required"):
                DeapiClient(base_url=TEST_BASE_URL)

    def test_default_config_values(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY)
        assert client._config.base_url == "https://api.deapi.ai"
        assert client._config.timeout == 30.0
        assert client._config.max_retries == 3
        assert client._config.api_version == "v1"

    def test_custom_config(self) -> None:
        client = DeapiClient(
            api_key=TEST_API_KEY,
            base_url="https://custom.deapi.ai",
            timeout=60.0,
            max_retries=5,
            api_version="v1",
        )
        assert client._config.base_url == "https://custom.deapi.ai"
        assert client._config.timeout == 60.0
        assert client._config.max_retries == 5

    def test_unsupported_api_version(self) -> None:
        with pytest.raises(DeapiError, match="Unsupported API version"):
            DeapiClient(api_key=TEST_API_KEY, api_version="v99")

    def test_context_manager(self) -> None:
        with DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL) as client:
            assert client._config.api_key == TEST_API_KEY

    def test_has_resources(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL)
        assert hasattr(client, "images")
        assert hasattr(client, "models")
        assert hasattr(client, "ocr")


class TestErrorMapping:
    @respx.mock
    def test_401_authentication_error(self, client: DeapiClient) -> None:
        respx.get(f"{TEST_BASE_URL}/api/v1/client/balance").mock(
            return_value=httpx.Response(401, json={"message": "Unauthenticated."})
        )
        with pytest.raises(AuthenticationError, match="Unauthenticated"):
            client.balance()

    @respx.mock
    def test_403_account_suspended(self, client: DeapiClient) -> None:
        respx.get(f"{TEST_BASE_URL}/api/v1/client/balance").mock(
            return_value=httpx.Response(403, json={"message": "Your account has been suspended."})
        )
        with pytest.raises(AccountSuspendedError, match="suspended"):
            client.balance()

    @respx.mock
    def test_404_not_found(self, client: DeapiClient) -> None:
        respx.get(f"{TEST_BASE_URL}/api/v1/client/balance").mock(
            return_value=httpx.Response(404, json={"message": "Not found."})
        )
        with pytest.raises(NotFoundError):
            client.balance()

    @respx.mock
    def test_422_validation_error(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2img").mock(
            return_value=httpx.Response(422, json={
                "message": "The selected model does not support Text To Image.",
                "errors": {"model": ["The selected model does not support Text To Image."]},
            })
        )
        with pytest.raises(ValidationError) as exc_info:
            client.images.generate(
                prompt="test", model="invalid", width=512, height=512, seed=1,
            )
        assert "model" in exc_info.value.errors

    @respx.mock
    def test_422_insufficient_balance(self, client: DeapiClient) -> None:
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2img").mock(
            return_value=httpx.Response(422, json={
                "message": "Not enough balance.",
                "errors": {"balance": ["Not enough balance."]},
            })
        )
        with pytest.raises(InsufficientBalanceError):
            client.images.generate(
                prompt="test", model="sdxl", width=512, height=512, seed=1,
            )

    @respx.mock
    def test_429_rate_limit(self, client: DeapiClient) -> None:
        respx.get(f"{TEST_BASE_URL}/api/v1/client/balance").mock(
            return_value=httpx.Response(
                429,
                json={"message": "Too Many Attempts."},
                headers={"Retry-After": "30", "X-RateLimit-Type": "minute"},
            )
        )
        # max_retries=3 default, all attempts get 429 → eventually raises
        client_no_retry = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, max_retries=0)
        with pytest.raises(RateLimitError) as exc_info:
            client_no_retry.balance()
        assert exc_info.value.retry_after == 30.0
        assert exc_info.value.limit_type == "minute"

    @respx.mock
    def test_500_server_error(self, client: DeapiClient) -> None:
        respx.get(f"{TEST_BASE_URL}/api/v1/client/balance").mock(
            return_value=httpx.Response(500, json={"message": "Server Error"})
        )
        client_no_retry = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, max_retries=0)
        with pytest.raises(ServerError):
            client_no_retry.balance()


class TestBalance:
    @respx.mock
    def test_get_balance(self, client: DeapiClient) -> None:
        respx.get(f"{TEST_BASE_URL}/api/v1/client/balance").mock(
            return_value=httpx.Response(200, json={"data": {"balance": 12.345}})
        )
        balance = client.balance()
        assert balance.balance == 12.345


class TestRetryLogic:
    @respx.mock
    def test_retry_on_429(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, max_retries=2)
        route = respx.get(f"{TEST_BASE_URL}/api/v1/client/balance")
        route.side_effect = [
            httpx.Response(429, json={"message": "Too Many Attempts."}, headers={"Retry-After": "0"}),
            httpx.Response(200, json={"data": {"balance": 5.0}}),
        ]
        balance = client.balance()
        assert balance.balance == 5.0
        assert route.call_count == 2

    @respx.mock
    def test_retry_on_500(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, max_retries=2)
        route = respx.get(f"{TEST_BASE_URL}/api/v1/client/balance")
        route.side_effect = [
            httpx.Response(500, json={"message": "Server Error"}),
            httpx.Response(200, json={"data": {"balance": 5.0}}),
        ]
        balance = client.balance()
        assert balance.balance == 5.0
        assert route.call_count == 2

    @respx.mock
    def test_max_retries_exhausted(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, max_retries=1)
        route = respx.get(f"{TEST_BASE_URL}/api/v1/client/balance")
        route.side_effect = [
            httpx.Response(500, json={"message": "Server Error"}),
            httpx.Response(500, json={"message": "Server Error"}),
        ]
        with pytest.raises(ServerError):
            client.balance()
        assert route.call_count == 2


class TestAsyncClient:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_balance(self, async_client: AsyncDeapiClient) -> None:
        respx.get(f"{TEST_BASE_URL}/api/v1/client/balance").mock(
            return_value=httpx.Response(200, json={"data": {"balance": 99.99}})
        )
        balance = await async_client.balance()
        assert balance.balance == 99.99

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        respx.get(f"{TEST_BASE_URL}/api/v1/client/balance").mock(
            return_value=httpx.Response(200, json={"data": {"balance": 1.0}})
        )
        async with AsyncDeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL) as client:
            balance = await client.balance()
            assert balance.balance == 1.0
