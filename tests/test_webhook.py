from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest

from deapi_sdk.webhook import (
    InvalidSignatureError,
    WebhookEvent,
    construct_event,
    verify_signature,
)

SECRET = "whsec_test_secret_123"

SAMPLE_PAYLOAD = json.dumps({
    "event": "job.completed",
    "delivery_id": "d1234567-abcd-efgh-ijkl-000000000001",
    "timestamp": "2024-01-15T10:30:45.000Z",
    "data": {
        "job_request_id": "c08a339c-73e5-4d67-a4d5-231302fbff9a",
        "status": "done",
        "previous_status": "processing",
        "job_type": "txt2img",
        "completed_at": "2024-01-15T10:30:45.000Z",
        "result_url": "https://storage.deapi.ai/results/test.png",
        "processing_time_ms": 45000,
    },
})


def _make_signature(payload: str, timestamp: str, secret: str = SECRET) -> str:
    """Create a valid signature for testing."""
    signed_content = f"{timestamp}.".encode() + payload.encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), signed_content, hashlib.sha256).hexdigest()
    return f"sha256={sig}"


class TestVerifySignature:
    def test_valid_signature(self) -> None:
        ts = str(int(time.time()))
        sig = _make_signature(SAMPLE_PAYLOAD, ts)
        verify_signature(
            payload=SAMPLE_PAYLOAD,
            signature=sig,
            timestamp=ts,
            secret=SECRET,
        )

    def test_valid_signature_with_bytes_payload(self) -> None:
        ts = str(int(time.time()))
        sig = _make_signature(SAMPLE_PAYLOAD, ts)
        verify_signature(
            payload=SAMPLE_PAYLOAD.encode("utf-8"),
            signature=sig,
            timestamp=ts,
            secret=SECRET,
        )

    def test_invalid_signature_raises(self) -> None:
        ts = str(int(time.time()))
        with pytest.raises(InvalidSignatureError):
            verify_signature(
                payload=SAMPLE_PAYLOAD,
                signature="sha256=invalid_hex_signature",
                timestamp=ts,
                secret=SECRET,
            )

    def test_wrong_secret_raises(self) -> None:
        ts = str(int(time.time()))
        sig = _make_signature(SAMPLE_PAYLOAD, ts, secret="wrong-secret")
        with pytest.raises(InvalidSignatureError):
            verify_signature(
                payload=SAMPLE_PAYLOAD,
                signature=sig,
                timestamp=ts,
                secret=SECRET,
            )

    def test_expired_timestamp_raises(self) -> None:
        old_ts = str(int(time.time()) - 600)
        sig = _make_signature(SAMPLE_PAYLOAD, old_ts)
        with pytest.raises(InvalidSignatureError, match="too old"):
            verify_signature(
                payload=SAMPLE_PAYLOAD,
                signature=sig,
                timestamp=old_ts,
                secret=SECRET,
                tolerance=300,
            )

    def test_invalid_timestamp_format_raises(self) -> None:
        sig = _make_signature(SAMPLE_PAYLOAD, "not-a-number")
        with pytest.raises(InvalidSignatureError, match="Invalid timestamp"):
            verify_signature(
                payload=SAMPLE_PAYLOAD,
                signature=sig,
                timestamp="not-a-number",
                secret=SECRET,
            )

    def test_zero_tolerance_skips_timestamp_check(self) -> None:
        old_ts = str(int(time.time()) - 99999)
        sig = _make_signature(SAMPLE_PAYLOAD, old_ts)
        verify_signature(
            payload=SAMPLE_PAYLOAD,
            signature=sig,
            timestamp=old_ts,
            secret=SECRET,
            tolerance=0,
        )

    def test_tampered_payload_raises(self) -> None:
        ts = str(int(time.time()))
        sig = _make_signature(SAMPLE_PAYLOAD, ts)
        tampered = SAMPLE_PAYLOAD.replace("done", "error")
        with pytest.raises(InvalidSignatureError):
            verify_signature(
                payload=tampered,
                signature=sig,
                timestamp=ts,
                secret=SECRET,
            )


class TestConstructEvent:
    def test_construct_event_parses_payload(self) -> None:
        ts = str(int(time.time()))
        sig = _make_signature(SAMPLE_PAYLOAD, ts)
        event = construct_event(
            payload=SAMPLE_PAYLOAD,
            signature=sig,
            timestamp=ts,
            secret=SECRET,
        )
        assert isinstance(event, WebhookEvent)
        assert event.event == "job.completed"
        assert event.type == "job.completed"
        assert event.delivery_id == "d1234567-abcd-efgh-ijkl-000000000001"
        assert event.data.job_request_id == "c08a339c-73e5-4d67-a4d5-231302fbff9a"
        assert event.data.status == "done"
        assert event.data.previous_status == "processing"
        assert event.data.job_type == "txt2img"
        assert event.data.result_url == "https://storage.deapi.ai/results/test.png"
        assert event.data.processing_time_ms == 45000

    def test_construct_event_with_bytes(self) -> None:
        ts = str(int(time.time()))
        payload_bytes = SAMPLE_PAYLOAD.encode("utf-8")
        sig = _make_signature(SAMPLE_PAYLOAD, ts)
        event = construct_event(
            payload=payload_bytes,
            signature=sig,
            timestamp=ts,
            secret=SECRET,
        )
        assert event.event == "job.completed"

    def test_construct_event_invalid_signature_raises(self) -> None:
        ts = str(int(time.time()))
        with pytest.raises(InvalidSignatureError):
            construct_event(
                payload=SAMPLE_PAYLOAD,
                signature="sha256=bad",
                timestamp=ts,
                secret=SECRET,
            )

    def test_construct_event_minimal_data(self) -> None:
        payload = json.dumps({
            "event": "job.processing",
            "delivery_id": "uuid-123",
            "timestamp": "2024-01-15T10:30:00.000Z",
            "data": {
                "job_request_id": "job-uuid",
                "status": "processing",
            },
        })
        ts = str(int(time.time()))
        sig = _make_signature(payload, ts)
        event = construct_event(
            payload=payload,
            signature=sig,
            timestamp=ts,
            secret=SECRET,
        )
        assert event.type == "job.processing"
        assert event.data.status == "processing"
        assert event.data.result_url is None
        assert event.data.processing_time_ms is None


class TestEventFields:
    def _construct(self, data: dict[str, object], event: str) -> WebhookEvent:
        envelope = {"event": event, "delivery_id": "d-1", "timestamp": "2026-01-01T00:00:00+00:00", "data": data}
        payload = json.dumps(envelope)
        ts = str(int(time.time()))
        return construct_event(payload=payload, signature=_make_signature(payload, ts), timestamp=ts, secret=SECRET)

    def test_failed_event_fields_are_typed(self) -> None:
        event = self._construct(
            {
                "job_request_id": "job-1",
                "status": "error",
                "previous_status": "processing",
                "job_type": "txt2img",
                "failed_at": "2026-01-01T00:00:05+00:00",
                "error_code": "WORKER_TIMEOUT",
                "error_reason": "TIMEOUT",
                "retryable": True,
                "refunded": True,
            },
            "job.failed",
        )
        data = event.data
        assert event.type == "job.failed"
        assert data.status == "error"
        assert data.failed_at == "2026-01-01T00:00:05+00:00"
        assert data.error_code == "WORKER_TIMEOUT"
        assert data.error_reason == "TIMEOUT"
        assert data.retryable is True
        assert data.refunded is True
        assert data.model_extra == {}

    def test_processing_event_started_at(self) -> None:
        event = self._construct(
            {"job_request_id": "job-1", "status": "processing", "previous_status": "pending",
             "started_at": "2026-01-01T00:00:01+00:00"},
            "job.processing",
        )
        assert event.data.started_at == "2026-01-01T00:00:01+00:00"
        assert event.data.error_code is None
        assert event.data.retryable is None

    def test_price_on_completed_and_failed(self) -> None:
        completed = self._construct(
            {"job_request_id": "job-1", "status": "done", "price": {"amount": 0.0123, "is_estimated": True}},
            "job.completed",
        )
        assert completed.data.price is not None
        assert completed.data.price.amount == 0.0123
        assert completed.data.price.is_estimated is True
        failed = self._construct(
            {"job_request_id": "job-1", "status": "error", "price": {"amount": 0, "is_estimated": False}},
            "job.failed",
        )
        assert failed.data.price is not None
        assert failed.data.price.amount == 0.0
        assert failed.data.price.is_estimated is False
        assert failed.data.model_extra == {}

    def test_price_missing_or_null(self) -> None:
        processing = self._construct({"job_request_id": "job-1", "status": "processing"}, "job.processing")
        assert processing.data.price is None
        null_price = self._construct({"job_request_id": "job-1", "status": "done", "price": None}, "job.completed")
        assert null_price.data.price is None

    def test_unknown_fields_still_kept(self) -> None:
        event = self._construct({"job_request_id": "job-1", "status": "done", "new_field": 1}, "job.completed")
        assert event.data.model_extra == {"new_field": 1}


class TestInvalidSignatureError:
    def test_is_deapi_error(self) -> None:
        from deapi_sdk._exceptions import DeapiError
        err = InvalidSignatureError()
        assert isinstance(err, DeapiError)

    def test_custom_message(self) -> None:
        err = InvalidSignatureError("custom message")
        assert str(err) == "custom message"
