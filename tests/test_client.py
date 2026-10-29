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
from deapi._client import BaseClient
from deapi._config import ClientConfig
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
        assert client._config.api_version == "v2"

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
        client_no_retry = DeapiClient(
            api_key=TEST_API_KEY, base_url=TEST_BASE_URL, max_retries=0, api_version="v1"
        )
        with pytest.raises(RateLimitError) as exc_info:
            client_no_retry.balance()
        assert exc_info.value.retry_after == 30.0
        assert exc_info.value.limit_type == "minute"

    @respx.mock
    def test_500_server_error(self, client: DeapiClient) -> None:
        respx.get(f"{TEST_BASE_URL}/api/v1/client/balance").mock(
            return_value=httpx.Response(500, json={"message": "Server Error"})
        )
        client_no_retry = DeapiClient(
            api_key=TEST_API_KEY, base_url=TEST_BASE_URL, max_retries=0, api_version="v1"
        )
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
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, max_retries=2, api_version="v1")
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
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, max_retries=2, api_version="v1")
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
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, max_retries=1, api_version="v1")
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
        async with AsyncDeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, api_version="v1") as client:
            balance = await client.balance()
            assert balance.balance == 1.0


def _base(max_retry_after: float = 60.0) -> BaseClient:
    return BaseClient(
        ClientConfig(api_key="k", max_retries=3, max_retry_after=max_retry_after)
    )


class TestRetryAfterCap:
    """A 429 Retry-After is honoured but never sleeps past the cap."""

    def test_default_cap_is_60s(self) -> None:
        assert ClientConfig(api_key="k").max_retry_after == 60.0

    def test_value_below_cap_is_honoured(self) -> None:
        wait = _base()._should_retry(RateLimitError("rl", retry_after=5.0), attempt=0)
        assert wait == 5.0

    def test_huge_value_is_capped(self) -> None:
        # e.g. a day-bucket Retry-After, or a far-future HTTP-date parsed to seconds
        wait = _base()._should_retry(RateLimitError("rl", retry_after=3600.0), attempt=0)
        assert wait == 60.0

    def test_cap_is_configurable(self) -> None:
        wait = _base(max_retry_after=45.0)._should_retry(
            RateLimitError("rl", retry_after=3600.0), attempt=0
        )
        assert wait == 45.0

    def test_floor_is_1s(self) -> None:
        wait = _base()._should_retry(RateLimitError("rl", retry_after=0.0), attempt=0)
        assert wait == 1.0


class TestInjectableHTTPClient:
    """An httpx client can be injected and shared; auth goes per-request."""

    def test_default_client_is_owned_and_closed(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL)
        assert client._http_client.owns_http_client is True
        underlying = client._http_client._http
        client.close()
        assert underlying.is_closed

    @respx.mock
    def test_injected_client_not_closed_and_auth_per_request(self) -> None:
        route = respx.get(f"{TEST_BASE_URL}/api/v1/client/balance").mock(
            return_value=httpx.Response(200, json={"data": {"balance": 1.0}})
        )
        injected = httpx.Client()  # no auth baked into the transport
        client = DeapiClient(
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            api_version="v1",
            http_client=injected,
        )
        assert client._http_client.owns_http_client is False
        client.balance()
        # auth travelled on the request, not on the shared transport
        assert route.calls.last.request.headers["Authorization"] == f"Bearer {TEST_API_KEY}"
        client.close()
        assert injected.is_closed is False  # injected client must survive close()
        injected.close()

    @respx.mock
    def test_injected_client_closed_when_owned(self) -> None:
        respx.get(f"{TEST_BASE_URL}/api/v1/client/balance").mock(
            return_value=httpx.Response(200, json={"data": {"balance": 1.0}})
        )
        injected = httpx.Client()
        client = DeapiClient(
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            api_version="v1",
            http_client=injected,
            owns_http_client=True,
        )
        assert client._http_client.owns_http_client is True
        client.close()
        assert injected.is_closed is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_injected_client_not_closed(self) -> None:
        respx.get(f"{TEST_BASE_URL}/api/v1/client/balance").mock(
            return_value=httpx.Response(200, json={"data": {"balance": 2.0}})
        )
        injected = httpx.AsyncClient()
        client = AsyncDeapiClient(
            api_key=TEST_API_KEY,
            base_url=TEST_BASE_URL,
            api_version="v1",
            http_client=injected,
        )
        assert client._http_client.owns_http_client is False
        await client.balance()
        await client.close()
        assert injected.is_closed is False
        await injected.aclose()
