from __future__ import annotations

from deapi_sdk._exceptions import (
    AccountSuspendedError,
    AuthenticationError,
    DeapiError,
    InsufficientBalanceError,
    JobTimeoutError,
    NetworkError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_deapi_error(self) -> None:
        exceptions = [
            AuthenticationError,
            AccountSuspendedError,
            NotFoundError,
            ValidationError,
            InsufficientBalanceError,
            RateLimitError,
            ServerError,
            JobTimeoutError,
            NetworkError,
        ]
        for exc_cls in exceptions:
            assert issubclass(exc_cls, DeapiError)

    def test_insufficient_balance_is_validation_error(self) -> None:
        assert issubclass(InsufficientBalanceError, ValidationError)

    def test_deapi_error_attributes(self) -> None:
        exc = DeapiError("test error", status_code=400, body={"key": "val"})
        assert str(exc) == "test error"
        assert exc.message == "test error"
        assert exc.status_code == 400
        assert exc.body == {"key": "val"}

    def test_validation_error_errors_dict(self) -> None:
        exc = ValidationError(
            "Validation failed",
            errors={"field": ["Error 1", "Error 2"]},
        )
        assert exc.errors == {"field": ["Error 1", "Error 2"]}

    def test_rate_limit_error_attributes(self) -> None:
        exc = RateLimitError(
            "Too many requests",
            retry_after=30.0,
            limit_type="minute",
        )
        assert exc.retry_after == 30.0
        assert exc.limit_type == "minute"

    def test_default_optional_attributes(self) -> None:
        exc = DeapiError("simple error")
        assert exc.status_code is None
        assert exc.body is None


class TestNewExceptions:
    def test_hierarchy(self) -> None:
        from deapi_sdk import (
            ActiveKeyLimitReachedError,
            InsufficientPermissionsError,
            IpNotAllowedError,
            KeyLimitExceededError,
            PayloadTooLargeError,
            PaymentRequiredError,
            PermissionDeniedError,
        )

        assert issubclass(KeyLimitExceededError, PaymentRequiredError)
        assert issubclass(AccountSuspendedError, PermissionDeniedError)
        assert issubclass(IpNotAllowedError, PermissionDeniedError)
        assert issubclass(InsufficientPermissionsError, PermissionDeniedError)
        assert not issubclass(IpNotAllowedError, AccountSuspendedError)
        assert issubclass(ActiveKeyLimitReachedError, ValidationError)
        assert issubclass(PayloadTooLargeError, DeapiError)

    def test_code_attribute(self) -> None:
        assert DeapiError("x").code is None
        assert DeapiError("x", code="foo").code == "foo"
        assert DeapiError("x", body={"code": "bar"}).code == "bar"

    def test_job_timeout_request_id(self) -> None:
        assert JobTimeoutError("t").request_id is None
        assert JobTimeoutError("t", request_id="r1").request_id == "r1"
