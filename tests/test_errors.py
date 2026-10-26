from __future__ import annotations

from email.utils import format_datetime
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from deapi import (
    AccountSuspendedError,
    ActiveKeyLimitReachedError,
    AsyncDeapiClient,
    AuthenticationError,
    DeapiClient,
    DeapiError,
    InsufficientBalanceError,
    InsufficientPermissionsError,
    IpNotAllowedError,
    KeyLimitExceededError,
    NetworkError,
    NotFoundError,
    PayloadTooLargeError,
    PaymentRequiredError,
    PermissionDeniedError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from tests.conftest import TEST_API_KEY, TEST_BASE_URL

BALANCE_URL = f"{TEST_BASE_URL}/api/v2/account/balance"
JOB_URL = f"{TEST_BASE_URL}/api/v2/txt2img"


def _v2(max_retries: int = 0) -> DeapiClient:
    return DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, api_version="v2", max_retries=max_retries)


class TestStatusCodeMapping:
    @pytest.mark.parametrize(
        ("status", "body", "exc_type", "code"),
        [
            (401, {"message": "Unauthenticated.", "code": "missing_key"}, AuthenticationError, "missing_key"),
            (401, {"message": "Unauthenticated.", "code": "invalid_key"}, AuthenticationError, "invalid_key"),
            (401, {"message": "Key expired.", "code": "key_expired"}, AuthenticationError, "key_expired"),
            (401, {"message": "Key revoked.", "code": "key_revoked"}, AuthenticationError, "key_revoked"),
            (402, {"message": "Limit.", "code": "key_limit_exceeded"}, KeyLimitExceededError, "key_limit_exceeded"),
            (402, {"message": "Pay up."}, PaymentRequiredError, None),
            (403, {"message": "Not allowed.", "code": "ip_not_allowed"}, IpNotAllowedError, "ip_not_allowed"),
            (403, {"message": "No.", "code": "insufficient_permissions"}, InsufficientPermissionsError,
             "insufficient_permissions"),
            (403, {"message": "Account suspended."}, AccountSuspendedError, None),
            (403, {"message": "Something new.", "code": "brand_new_code"}, PermissionDeniedError, "brand_new_code"),
            (404, {"message": "Not found."}, NotFoundError, None),
            (413, {"message": "Too large."}, PayloadTooLargeError, None),
            (422, {"message": "Low balance.", "code": "insufficient_balance"}, InsufficientBalanceError,
             "insufficient_balance"),
            (422, {"message": "Invalid.", "errors": {"balance": ["low"]}}, InsufficientBalanceError, None),
            (422, {"message": "Too many keys.", "code": "active_key_limit_reached"}, ActiveKeyLimitReachedError,
             "active_key_limit_reached"),
            (422, {"message": "Invalid.", "errors": {"prompt": ["required"]}}, ValidationError, None),
            (429, {"message": "Slow down.", "code": "rate_limited"}, RateLimitError, "rate_limited"),
            (500, {"message": "Boom."}, ServerError, None),
            (418, {"message": "Teapot."}, DeapiError, None),
        ],
    )
    @respx.mock
    def test_mapping(self, status: int, body: dict, exc_type: type, code: str | None) -> None:  # type: ignore[type-arg]
        respx.get(BALANCE_URL).mock(return_value=httpx.Response(status, json=body))
        with _v2() as client, pytest.raises(DeapiError) as exc_info:
            client.balance()
        exc = exc_info.value
        assert type(exc) is exc_type
        assert exc.status_code == status
        assert exc.code == code
        assert exc.message == body["message"]
        assert exc.body == body

    @respx.mock
    def test_403_ip_not_allowed_is_not_account_suspended(self) -> None:
        respx.get(BALANCE_URL).mock(
            return_value=httpx.Response(403, json={"message": "IP not allowed.", "code": "ip_not_allowed"})
        )
        with _v2() as client, pytest.raises(PermissionDeniedError) as exc_info:
            client.balance()
        assert not isinstance(exc_info.value, AccountSuspendedError)
        assert isinstance(exc_info.value, IpNotAllowedError)

    @respx.mock
    def test_403_insufficient_permissions_is_not_account_suspended(self) -> None:
        respx.get(BALANCE_URL).mock(
            return_value=httpx.Response(403, json={"message": "Forbidden.", "code": "insufficient_permissions"})
        )
        with _v2() as client, pytest.raises(PermissionDeniedError) as exc_info:
            client.balance()
        assert not isinstance(exc_info.value, AccountSuspendedError)
        assert isinstance(exc_info.value, InsufficientPermissionsError)

    @respx.mock
    def test_402_key_limit_exceeded_dedicated_exception(self) -> None:
        respx.get(BALANCE_URL).mock(
            return_value=httpx.Response(
                402,
                json={
                    "message": "Key limit exceeded.",
                    "code": "key_limit_exceeded",
                    "limit": 10,
                    "used": 10.5,
                    "remaining": 0,
                    "resets_at": "2026-11-01T00:00:00Z",
                },
            )
        )
        with _v2() as client, pytest.raises(KeyLimitExceededError) as exc_info:
            client.balance()
        exc = exc_info.value
        assert isinstance(exc, PaymentRequiredError)
        assert (exc.limit, exc.used, exc.remaining) == (10.0, 10.5, 0.0)
        assert exc.resets_at == "2026-11-01T00:00:00Z"

    @respx.mock
    def test_402_key_limit_exceeded_null_reset(self) -> None:
        respx.get(BALANCE_URL).mock(
            return_value=httpx.Response(402, json={"message": "x", "code": "key_limit_exceeded", "resets_at": None})
        )
        with _v2() as client, pytest.raises(KeyLimitExceededError) as exc_info:
            client.balance()
        assert exc_info.value.resets_at is None
        assert exc_info.value.limit is None

    @respx.mock
    def test_403_without_code_still_caught_as_account_suspended(self) -> None:
        respx.get(BALANCE_URL).mock(return_value=httpx.Response(403, json={"message": "Account suspended."}))
        with _v2() as client:
            try:
                client.balance()
            except AccountSuspendedError as exc:
                assert isinstance(exc, PermissionDeniedError)
            else:  # pragma: no cover
                pytest.fail("expected AccountSuspendedError")

    def test_insufficient_balance_still_validation_error(self) -> None:
        assert issubclass(InsufficientBalanceError, ValidationError)
        assert issubclass(ActiveKeyLimitReachedError, ValidationError)


class TestBodyRobustness:
    @respx.mock
    def test_bare_error_field_used_as_message(self) -> None:
        respx.get(BALANCE_URL).mock(return_value=httpx.Response(404, json={"error": "No such thing"}))
        with _v2() as client, pytest.raises(NotFoundError) as exc_info:
            client.balance()
        assert exc_info.value.message == "No such thing"

    @pytest.mark.parametrize("payload", [["a", "b"], "oops", 42, None])
    @respx.mock
    def test_non_dict_json_body_does_not_crash(self, payload: Any) -> None:
        respx.get(BALANCE_URL).mock(return_value=httpx.Response(403, json=payload))
        with _v2() as client, pytest.raises(AccountSuspendedError) as exc_info:
            client.balance()
        assert exc_info.value.code is None

    @respx.mock
    def test_non_json_error_body_uses_text(self) -> None:
        respx.get(BALANCE_URL).mock(return_value=httpx.Response(502, text="Bad gateway"))
        with _v2() as client, pytest.raises(ServerError) as exc_info:
            client.balance()
        assert exc_info.value.message == "Bad gateway"

    @respx.mock
    def test_empty_error_body_uses_http_status(self) -> None:
        respx.get(BALANCE_URL).mock(return_value=httpx.Response(418))
        with _v2() as client, pytest.raises(DeapiError) as exc_info:
            client.balance()
        assert exc_info.value.message == "HTTP 418"

    @respx.mock
    def test_422_errors_as_list(self) -> None:
        respx.get(BALANCE_URL).mock(
            return_value=httpx.Response(422, json={"message": "Bad", "errors": ["one", "two"]})
        )
        with _v2() as client, pytest.raises(ValidationError) as exc_info:
            client.balance()
        assert exc_info.value.errors == {"_": ["one", "two"]}

    @respx.mock
    def test_2xx_non_json_raises_deapi_error(self) -> None:
        respx.get(BALANCE_URL).mock(return_value=httpx.Response(200, text="<html>nope</html>"))
        with _v2() as client, pytest.raises(DeapiError) as exc_info:
            client.balance()
        assert "Invalid JSON" in exc_info.value.message


class TestRetryAfter:
    @pytest.mark.parametrize(
        ("header", "expected"),
        [("30", 30.0), ("1.5", 1.5), ("-4", 0.0), ("soon", 0.0), ("", 0.0), ("nan", 0.0)],
    )
    @respx.mock
    def test_numeric_and_garbage(self, header: str, expected: float) -> None:
        respx.get(BALANCE_URL).mock(
            return_value=httpx.Response(429, json={"message": "x"}, headers={"Retry-After": header})
        )
        with _v2() as client, pytest.raises(RateLimitError) as exc_info:
            client.balance()
        assert exc_info.value.retry_after == expected

    @respx.mock
    def test_missing_header(self) -> None:
        respx.get(BALANCE_URL).mock(return_value=httpx.Response(429, json={"message": "x"}))
        with _v2() as client, pytest.raises(RateLimitError) as exc_info:
            client.balance()
        assert exc_info.value.retry_after == 0.0
        assert exc_info.value.limit_type == ""

    @respx.mock
    def test_http_date(self) -> None:
        when = datetime.now(timezone.utc) + timedelta(seconds=60)
        respx.get(BALANCE_URL).mock(
            return_value=httpx.Response(
                429,
                json={"message": "x"},
                headers={"Retry-After": format_datetime(when, usegmt=True), "X-RateLimit-Type": "daily"},
            )
        )
        with _v2() as client, pytest.raises(RateLimitError) as exc_info:
            client.balance()
        assert 50 <= exc_info.value.retry_after <= 61
        assert exc_info.value.limit_type == "daily"

    @respx.mock
    def test_http_date_in_past_is_zero(self) -> None:
        respx.get(BALANCE_URL).mock(
            return_value=httpx.Response(
                429, json={"message": "x"}, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}
            )
        )
        with _v2() as client, pytest.raises(RateLimitError) as exc_info:
            client.balance()
        assert exc_info.value.retry_after == 0.0


class TestRetryPolicy:
    """POST/PATCH/DELETE must not be retried unless the request provably never reached processing."""

    @respx.mock
    def test_get_retried_on_500(self) -> None:
        route = respx.get(BALANCE_URL)
        route.side_effect = [httpx.Response(500, json={"message": "x"}), httpx.Response(200, json={"data": {"balance": 1}})]
        with _v2(max_retries=2) as client, patch("deapi._client.time.sleep") as sleep:
            assert client.balance().balance == 1
        assert route.call_count == 2
        sleep.assert_called_once()

    @respx.mock
    def test_get_retried_on_read_timeout(self) -> None:
        route = respx.get(BALANCE_URL)
        route.side_effect = [httpx.ReadTimeout("slow"), httpx.Response(200, json={"data": {"balance": 1}})]
        with _v2(max_retries=2) as client, patch("deapi._client.time.sleep"):
            client.balance()
        assert route.call_count == 2

    @respx.mock
    def test_post_not_retried_on_500(self) -> None:
        route = respx.post(JOB_URL).mock(return_value=httpx.Response(500, json={"message": "x"}))
        with _v2(max_retries=3) as client, patch("deapi._client.time.sleep") as sleep:
            with pytest.raises(ServerError):
                client._http_client.post(JOB_URL, json={})  # noqa: SLF001
        assert route.call_count == 1
        sleep.assert_not_called()

    @pytest.mark.parametrize("exc", [httpx.ReadTimeout("r"), httpx.WriteTimeout("w"), httpx.PoolTimeout("p")])
    @respx.mock
    def test_post_not_retried_on_non_connect_timeout(self, exc: Exception) -> None:
        route = respx.post(JOB_URL).mock(side_effect=exc)
        with _v2(max_retries=3) as client, patch("deapi._client.time.sleep"):
            with pytest.raises(NetworkError):
                client._http_client.post(JOB_URL, json={})  # noqa: SLF001
        assert route.call_count == 1

    @respx.mock
    def test_post_retried_on_429(self) -> None:
        route = respx.post(JOB_URL)
        route.side_effect = [
            httpx.Response(429, json={"message": "x"}, headers={"Retry-After": "0"}),
            httpx.Response(200, json={"data": {"request_id": "abc"}}),
        ]
        with _v2(max_retries=2) as client, patch("deapi._client.time.sleep"):
            assert client._http_client.post(JOB_URL, json={}) == {"data": {"request_id": "abc"}}  # noqa: SLF001
        assert route.call_count == 2

    @pytest.mark.parametrize("exc", [httpx.ConnectError("refused"), httpx.ConnectTimeout("slow connect")])
    @respx.mock
    def test_post_retried_on_connect_failure(self, exc: Exception) -> None:
        route = respx.post(JOB_URL)
        route.side_effect = [exc, httpx.Response(200, json={"ok": True})]
        with _v2(max_retries=2) as client, patch("deapi._client.time.sleep"):
            assert client._http_client.post(JOB_URL, json={}) == {"ok": True}  # noqa: SLF001
        assert route.call_count == 2

    @respx.mock
    def test_patch_and_delete_not_retried_on_500(self) -> None:
        patch_route = respx.patch(JOB_URL).mock(return_value=httpx.Response(503, json={"message": "x"}))
        delete_route = respx.delete(JOB_URL).mock(return_value=httpx.Response(503, json={"message": "x"}))
        with _v2(max_retries=3) as client, patch("deapi._client.time.sleep"):
            with pytest.raises(ServerError):
                client._http_client.patch(JOB_URL, json={})  # noqa: SLF001
            with pytest.raises(ServerError):
                client._http_client.delete(JOB_URL)  # noqa: SLF001
        assert patch_route.call_count == 1
        assert delete_route.call_count == 1

    @respx.mock
    def test_delete_empty_body_returns_empty_dict(self) -> None:
        respx.delete(JOB_URL).mock(return_value=httpx.Response(204))
        with _v2() as client:
            assert client._http_client.delete(JOB_URL) == {}  # noqa: SLF001

    @respx.mock
    def test_patch_returns_json(self) -> None:
        route = respx.patch(JOB_URL).mock(return_value=httpx.Response(200, json={"data": {"x": 1}}))
        with _v2() as client:
            assert client._http_client.patch(JOB_URL, json={"a": 1}) == {"data": {"x": 1}}  # noqa: SLF001
        assert route.calls[0].request.content == b'{"a":1}'


class TestAsyncParity:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_403_ip_not_allowed(self, async_client_v2: AsyncDeapiClient) -> None:
        respx.get(BALANCE_URL).mock(
            return_value=httpx.Response(403, json={"message": "x", "code": "ip_not_allowed"})
        )
        with pytest.raises(IpNotAllowedError):
            await async_client_v2.balance()

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_402_key_limit(self, async_client_v2: AsyncDeapiClient) -> None:
        respx.get(BALANCE_URL).mock(
            return_value=httpx.Response(402, json={"message": "x", "code": "key_limit_exceeded", "limit": 5})
        )
        with pytest.raises(KeyLimitExceededError) as exc_info:
            await async_client_v2.balance()
        assert exc_info.value.limit == 5.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_post_not_retried_on_500_but_retried_on_connect_error(self) -> None:
        async def fake_sleep(_: float) -> None:
            return None

        client = AsyncDeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, api_version="v2", max_retries=3)
        try:
            with patch("deapi._client.asyncio.sleep", fake_sleep):
                route = respx.post(JOB_URL).mock(return_value=httpx.Response(500, json={"message": "x"}))
                with pytest.raises(ServerError):
                    await client._http_client.post(JOB_URL, json={})  # noqa: SLF001
                assert route.call_count == 1

                route2 = respx.post(f"{JOB_URL}2")
                route2.side_effect = [httpx.ConnectError("no"), httpx.Response(200, json={"ok": 1})]
                assert await client._http_client.post(f"{JOB_URL}2", json={}) == {"ok": 1}  # noqa: SLF001
                assert route2.call_count == 2

                route3 = respx.post(f"{JOB_URL}3").mock(side_effect=httpx.ReadTimeout("t"))
                with pytest.raises(NetworkError):
                    await client._http_client.post(f"{JOB_URL}3", json={})  # noqa: SLF001
                assert route3.call_count == 1

                route4 = respx.patch(f"{JOB_URL}4").mock(return_value=httpx.Response(204))
                assert await client._http_client.patch(f"{JOB_URL}4", json={}) == {}  # noqa: SLF001
                assert route4.call_count == 1
        finally:
            await client.close()
