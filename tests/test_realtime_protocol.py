from __future__ import annotations

import json
from urllib.parse import parse_qsl, urlsplit

import pytest

from deapi_sdk._realtime import _protocol as protocol
from deapi_sdk._realtime._protocol import CloseAction
from deapi_sdk._types import JobStatus
from deapi_sdk.types.common import JobStatusEvent


class TestBuildUrl:
    @pytest.mark.parametrize("scheme", ["ws", "wss"])
    def test_schemes(self, scheme: str) -> None:
        url = protocol.build_url(f"{scheme}://example.test", "key", "1.2.3")
        assert url.startswith(f"{scheme}://example.test/app/key?")

    def test_exact_path_and_query(self) -> None:
        url = protocol.build_url("wss://example.test", "my-key", "1.2.3")
        assert url == "wss://example.test/app/my-key?protocol=7&client=deapi-python&version=1.2.3"
        parts = urlsplit(url)
        assert parts.path == "/app/my-key"
        assert dict(parse_qsl(parts.query)) == {"protocol": "7", "client": "deapi-python", "version": "1.2.3"}

    def test_trailing_slash_stripped(self) -> None:
        a = protocol.build_url("wss://example.test/", "key", "1")
        b = protocol.build_url("wss://example.test", "key", "1")
        assert a == b
        assert "//app" not in a.split("://", 1)[1]

    def test_port_preserved(self) -> None:
        assert protocol.build_url("ws://127.0.0.1:6001", "key", "1").startswith("ws://127.0.0.1:6001/app/key?")

    @pytest.mark.parametrize("bad", ["https://example.test", "http://example.test", "example.test", "wss://", ""])
    def test_invalid_ws_url(self, bad: str) -> None:
        with pytest.raises(ValueError, match="ws_url"):
            protocol.build_url(bad, "key", "1")

    def test_empty_key(self) -> None:
        with pytest.raises(ValueError, match="app_key"):
            protocol.build_url("wss://example.test", "", "1")

    def test_key_is_quoted(self) -> None:
        assert "/app/a%2Fb?" in protocol.build_url("wss://example.test", "a/b", "1")


class TestChannel:
    def test_channel_for(self) -> None:
        assert protocol.channel_for("test-client-1") == "private-client.test-client-1"


class TestDecode:
    def test_double_encoded_data(self) -> None:
        frame = json.dumps({"event": "x", "data": json.dumps({"a": 1})})
        msg = protocol.decode(frame)
        assert msg == protocol.Message(event="x", data={"a": 1}, channel=None)

    def test_object_data(self) -> None:
        msg = protocol.decode(json.dumps({"event": "x", "data": {"a": 1}}))
        assert msg is not None
        assert msg.data == {"a": 1}

    def test_non_json_string_data_kept(self) -> None:
        msg = protocol.decode(json.dumps({"event": "x", "data": "hello world"}))
        assert msg is not None
        assert msg.data == "hello world"

    def test_missing_data(self) -> None:
        msg = protocol.decode(json.dumps({"event": "x"}))
        assert msg is not None
        assert msg.data is None

    def test_channel(self) -> None:
        msg = protocol.decode(json.dumps({"event": "x", "channel": "private-client.1", "data": "{}"}))
        assert msg is not None
        assert msg.channel == "private-client.1"
        assert msg.data == {}

    def test_non_string_channel_ignored(self) -> None:
        msg = protocol.decode(json.dumps({"event": "x", "channel": 5}))
        assert msg is not None
        assert msg.channel is None

    def test_bytes_frame(self) -> None:
        msg = protocol.decode(b'{"event":"x","data":"{}"}')
        assert msg is not None
        assert msg.event == "x"

    @pytest.mark.parametrize("garbage", ["not json", "", "[1, 2]", "42", "null", '"str"', "{"])
    def test_garbage_returns_none(self, garbage: str) -> None:
        assert protocol.decode(garbage) is None

    @pytest.mark.parametrize("frame", ['{"data": "{}"}', '{"event": 5}', '{"event": null}'])
    def test_missing_or_bad_event_returns_none(self, frame: str) -> None:
        assert protocol.decode(frame) is None


class TestFrames:
    def test_subscribe_frame(self) -> None:
        raw = json.loads(protocol.subscribe_frame("private-client.1", "key:sig"))
        assert raw == {"event": "pusher:subscribe", "data": {"channel": "private-client.1", "auth": "key:sig"}}

    def test_ping_pong_frames(self) -> None:
        assert json.loads(protocol.PING_FRAME) == {"event": "pusher:ping", "data": {}}
        assert json.loads(protocol.PONG_FRAME) == {"event": "pusher:pong", "data": {}}

    def test_encode_is_compact(self) -> None:
        assert protocol.encode("e", {"a": 1}) == '{"event":"e","data":{"a":1}}'


class TestCloseAction:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (4000, CloseAction.FATAL),
            (4001, CloseAction.FATAL),
            (4009, CloseAction.FATAL),
            (4099, CloseAction.FATAL),
            (4100, CloseAction.BACKOFF),
            (4101, CloseAction.BACKOFF),
            (4199, CloseAction.BACKOFF),
            (4200, CloseAction.RECONNECT),
            (4201, CloseAction.RECONNECT),
            (4299, CloseAction.RECONNECT),
            (4300, CloseAction.BACKOFF),
            (3999, CloseAction.BACKOFF),
            (1000, CloseAction.BACKOFF),
            (1001, CloseAction.BACKOFF),
            (1006, CloseAction.BACKOFF),
            (None, CloseAction.BACKOFF),
        ],
    )
    def test_table(self, code: int | None, expected: CloseAction) -> None:
        assert protocol.close_action(code) is expected


class TestBackoffDelay:
    @pytest.mark.parametrize("attempt", range(0, 8))
    def test_bounds(self, attempt: int) -> None:
        ceiling = min(30.0, 2.0**attempt)
        low = protocol.backoff_delay(attempt, rand=lambda: 0.0)
        high = protocol.backoff_delay(attempt, rand=lambda: 1.0)
        assert low == pytest.approx(max(1.0, ceiling * 0.5))
        assert high == pytest.approx(ceiling)
        assert 1.0 <= low <= high

    def test_cap(self) -> None:
        assert protocol.backoff_delay(20, rand=lambda: 1.0) == 30.0
        assert protocol.backoff_delay(20, cap=5.0, rand=lambda: 1.0) == 5.0

    def test_never_below_base(self) -> None:
        for attempt in range(-2, 12):
            assert protocol.backoff_delay(attempt, rand=lambda: 0.0) >= 1.0
        assert protocol.backoff_delay(0, base=0.25, rand=lambda: 0.0) == 0.25

    def test_default_rand_within_bounds(self) -> None:
        for _ in range(50):
            assert 1.0 <= protocol.backoff_delay(3) <= 8.0

    def test_negative_attempt_treated_as_zero(self) -> None:
        assert protocol.backoff_delay(-5, rand=lambda: 1.0) == 1.0


class TestErrorHelpers:
    def test_error_code(self) -> None:
        assert protocol.error_code({"code": 4001, "message": "x"}) == 4001

    @pytest.mark.parametrize("data", [None, "x", {}, {"code": "4001"}, {"code": True}, {"code": 4.5}, [4001]])
    def test_error_code_missing(self, data: object) -> None:
        assert protocol.error_code(data) is None

    def test_error_text_preference(self) -> None:
        assert protocol.error_text({"message": "m", "error": "e", "type": "t"}) == "m"
        assert protocol.error_text({"error": "e", "type": "t"}) == "e"
        assert protocol.error_text({"type": "t"}) == "t"

    def test_error_text_string(self) -> None:
        assert protocol.error_text("plain") == "plain"

    @pytest.mark.parametrize("data", [None, {}, "", {"message": ""}, {"message": 5}, 5])
    def test_error_text_fallback(self, data: object) -> None:
        assert protocol.error_text(data) == "no details"


class TestJobStatusEvent:
    def test_progress_string_parsed(self) -> None:
        ev = JobStatusEvent.model_validate({"request_id": "req-1", "status": "processing", "progress": "45.50"})
        assert ev.progress == 45.5
        assert isinstance(ev.progress, float)

    @pytest.mark.parametrize("raw", ["", "   ", "garbage", "12%"])
    def test_progress_unparseable_is_none(self, raw: str) -> None:
        ev = JobStatusEvent.model_validate({"request_id": "req-1", "status": "processing", "progress": raw})
        assert ev.progress is None

    def test_progress_none_and_missing(self) -> None:
        assert JobStatusEvent.model_validate({"request_id": "r", "status": "done", "progress": None}).progress is None
        assert JobStatusEvent.model_validate({"request_id": "r", "status": "done"}).progress is None

    @pytest.mark.parametrize(("raw", "expected"), [(10, 10.0), (55.5, 55.5), (0, 0.0)])
    def test_numeric_passthrough(self, raw: float, expected: float) -> None:
        ev = JobStatusEvent.model_validate({"request_id": "r", "status": "processing", "progress": raw})
        assert ev.progress == expected

    @pytest.mark.parametrize(
        ("raw", "member"),
        [
            ("processing", JobStatus.PROCESSING),
            ("done", JobStatus.DONE),
        ],
    )
    def test_known_status_is_enum_member(self, raw: str, member: JobStatus) -> None:
        ev = JobStatusEvent.model_validate({"request_id": "r", "status": raw})
        assert ev.status is member

    def test_in_progress_is_not_a_known_status(self) -> None:
        # Progress is reported as repeated "processing" events; "in_progress" is not part of the API.
        ev = JobStatusEvent.model_validate({"request_id": "r", "status": "in_progress"})
        assert ev.status == "in_progress"
        assert not isinstance(ev.status, JobStatus)

    def test_unknown_status_kept_as_str(self) -> None:
        ev = JobStatusEvent.model_validate({"request_id": "r", "status": "queued_for_gpu"})
        assert ev.status == "queued_for_gpu"
        assert not isinstance(ev.status, JobStatus)

    def test_extra_fields_allowed(self) -> None:
        ev = JobStatusEvent.model_validate({"request_id": "r", "status": "done", "future_field": {"a": 1}})
        assert ev.model_extra == {"future_field": {"a": 1}}

    def test_optional_fields(self) -> None:
        ev = JobStatusEvent.model_validate(
            {"request_id": "r", "status": "done", "preview": "p", "result_url": "https://result.test/x.png"}
        )
        assert ev.preview == "p"
        assert ev.result_url == "https://result.test/x.png"

    def test_request_id_required(self) -> None:
        with pytest.raises(ValueError):  # noqa: PT011
            JobStatusEvent.model_validate({"status": "done"})
