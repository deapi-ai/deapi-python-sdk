from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest

from deapi.webhook import (
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
    signed_content = f"{timestamp}.".encode("utf-8") + payload.encode("utf-8")
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


class TestInvalidSignatureError:
    def test_is_deapi_error(self) -> None:
        from deapi._exceptions import DeapiError
        err = InvalidSignatureError()
        assert isinstance(err, DeapiError)

    def test_custom_message(self) -> None:
        err = InvalidSignatureError("custom message")
        assert str(err) == "custom message"
