"""v2 transcription: new parameters, JSON vs plain-text results, get_transcript."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from deapi import AsyncDeapiClient, DeapiClient
from deapi.transcript import Transcript, parse_transcript, transcript_output_format
from deapi.types.common import JobResult

V2 = "https://test.deapi.ai/api/v2"
JOB = {"data": {"request_id": "tr-1"}}
PRICE = {"data": {"price": 0.07}}
STORAGE = "https://storage.example.com/results/tr-1.json"

CT2_JSON = {
    "text": "Hello there. Hi!",
    "language": "en",
    "ts_level": "word",
    "diarization_available": True,
    "segments": [
        {
            "start": 0.0,
            "end": 1.5,
            "text": "Hello there.",
            "speaker": "SPEAKER_00",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.6, "score": 0.98, "speaker": "SPEAKER_00"},
                {"word": "there.", "start": 0.7, "end": 1.5, "score": 0.91},
            ],
        },
        {"start": 1.6, "end": 2.0, "text": "Hi!", "speaker": "SPEAKER_01", "avg_logprob": -0.2},
    ],
}


def done(**kw: object) -> JobResult:
    return JobResult.model_validate({"status": "done", **kw})


# --- parse_transcript ---


class TestParseTranscript:
    def test_ct2_json_with_words_and_speakers(self) -> None:
        t = parse_transcript(json.dumps(CT2_JSON), output_format="json")
        assert t.format == "json"
        assert t.text == "Hello there. Hi!"
        assert t.language == "en"
        assert t.ts_level == "word"
        assert t.diarization_available is True
        assert len(t.segments) == 2
        assert t.segments[0].speaker == "SPEAKER_00"
        assert t.segments[0].words[1].word == "there."
        assert t.segments[0].words[0].score == 0.98
        assert t.segments[1].words is None
        assert t.segments[1].avg_logprob == -0.2  # extra fields kept
        assert json.loads(t.raw)["language"] == "en"

    def test_json_without_timestamps(self) -> None:
        t = parse_transcript('{"text": "just text", "language": "pl", "segments": []}')
        assert t.format == "json"
        assert t.text == "just text"
        assert t.segments == []
        assert t.ts_level is None

    def test_dict_input(self) -> None:
        t = parse_transcript(CT2_JSON)
        assert t.format == "json" and len(t.segments) == 2

    def test_plain_text_whisper(self) -> None:
        raw = "[0:00 - 0:05] Good morning.\n[0:05 - 0:09] Shall we start?"
        t = parse_transcript(raw)
        assert t.format == "text"
        assert t.text == raw and t.raw == raw
        assert t.segments == [] and t.language is None

    def test_forced_text_never_decodes(self) -> None:
        raw = json.dumps(CT2_JSON)
        t = parse_transcript(raw, output_format="text")
        assert t.format == "text" and t.text == raw

    @pytest.mark.parametrize("raw", ['{"text": "cut off', "{not json}", '{"no_text": 1}', '["a"]', "", '{"text": 5}'])
    def test_malformed_falls_back_to_text(self, raw: str) -> None:
        t = parse_transcript(raw, output_format="json")
        assert t.format == "text"
        assert t.text == raw

    def test_none_is_empty_text(self) -> None:
        t = parse_transcript(None)
        assert t.format == "text" and t.text == ""

    def test_bad_segment_skipped(self) -> None:
        t = parse_transcript({"text": "x", "segments": ["oops", {"text": "ok", "start": 1}]})
        assert [s.text for s in t.segments] == ["ok"]

    def test_empty_transcript_is_valid(self) -> None:
        t = parse_transcript('{"text": "", "segments": []}')
        assert t.format == "json" and t.text == "" and t.segments == []

    def test_output_format_from_card(self) -> None:
        assert transcript_output_format({"info": {"limits": {"output_format": "json"}}}) == "json"
        assert transcript_output_format({"info": {"limits": {"output_format": None}}}) is None
        assert transcript_output_format({"slug": "x"}) is None


# --- create / create_price parameters ---


class TestCreateParams:
    @respx.mock
    def test_url_json_payload(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/audio/transcriptions").mock(return_value=httpx.Response(200, json=JOB))
        client_v2.transcription.create(
            model="WhisperLargeV3Ct2", source_url="https://youtu.be/x", ts_level="word", diarize=True,
            lang="pl", include_metadata=True, webhook_url="https://hook.example.com/h", webhook_secret="s" * 32,
        )
        body = json.loads(route.calls.last.request.content)
        assert body == {
            "source_url": "https://youtu.be/x", "model": "WhisperLargeV3Ct2", "include_ts": True,
            "ts_level": "word", "diarize": True, "lang": "pl", "include_metadata": True,
            "webhook_url": "https://hook.example.com/h", "webhook_secret": "s" * 32,
        }

    @respx.mock
    def test_unset_fields_omitted(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/audio/transcriptions").mock(return_value=httpx.Response(200, json=JOB))
        client_v2.transcription.create(model="WhisperLargeV3", source_url="https://x.test/a.mp3")
        assert json.loads(route.calls.last.request.content) == {
            "source_url": "https://x.test/a.mp3", "model": "WhisperLargeV3",
        }

    @respx.mock
    def test_explicit_include_ts_false_kept(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/audio/transcriptions").mock(return_value=httpx.Response(200, json=JOB))
        client_v2.transcription.create(model="m", source_url="https://x.test/a", include_ts=False)
        assert json.loads(route.calls.last.request.content)["include_ts"] is False

    @respx.mock
    def test_file_multipart_bools(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/audio/transcriptions").mock(return_value=httpx.Response(200, json=JOB))
        client_v2.transcription.create(
            model="WhisperLargeV3Ct2", source_file=b"audio", diarize=True, lang="en", return_result_in_response=False,
        )
        req = route.calls.last.request
        assert "multipart/form-data" in req.headers["content-type"]
        body = req.content.decode("latin-1")
        for name, value in [("include_ts", "1"), ("diarize", "1"), ("lang", "en"), ("return_result_in_response", "0")]:
            assert f'name="{name}"\r\n\r\n{value}\r\n' in body
        assert 'name="ts_level"' not in body and 'name="include_metadata"' not in body

    @respx.mock
    def test_include_metadata_with_file_allowed(self, client_v2: DeapiClient) -> None:
        # Spec: a file upload simply returns null metadata and is never charged for the flag.
        job = respx.post(f"{V2}/audio/transcriptions").mock(return_value=httpx.Response(200, json=JOB))
        price = respx.post(f"{V2}/audio/transcriptions/price").mock(return_value=httpx.Response(200, json=PRICE))
        client_v2.transcription.create(model="m", source_file=b"a", include_metadata=True)
        client_v2.transcription.create_price(model="m", source_file=b"a", include_metadata=True)
        for route in (job, price):
            assert 'name="include_metadata"\r\n\r\n1\r\n' in route.calls.last.request.content.decode("latin-1")

    @respx.mock
    def test_price_sends_lang(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/audio/transcriptions/price").mock(return_value=httpx.Response(200, json=PRICE))
        client_v2.transcription.create_price(model="m", source_url="https://x.test/a", lang="pl")
        assert json.loads(route.calls.last.request.content) == {
            "model": "m", "lang": "pl", "source_url": "https://x.test/a",
        }

    def test_webhook_secret_validation(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError, match="requires webhook_url"):
            client_v2.transcription.create(model="m", source_url="https://x.test/a", webhook_secret="s" * 32)
        with pytest.raises(ValueError, match="32-255"):
            client_v2.transcription.create(
                model="m", source_url="https://x.test/a", webhook_url="https://h.test/x", webhook_secret="short",
            )

    @respx.mock
    def test_legacy_wrapper_forwards_new_options(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/audio/transcriptions").mock(return_value=httpx.Response(200, json=JOB))
        with pytest.warns(DeprecationWarning):
            client_v2.transcription.from_audio_url(
                audio_url="https://x.test/a", include_ts=True, model="m", ts_level="segment", lang="en",
            )
        body = json.loads(route.calls.last.request.content)
        assert body["ts_level"] == "segment" and body["lang"] == "en"

    @respx.mock
    def test_price_params(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/audio/transcriptions/price").mock(return_value=httpx.Response(200, json=PRICE))
        price = client_v2.transcription.create_price(
            model="WhisperLargeV3Ct2", duration_seconds=240, platform="tiktok", ts_level="word", include_metadata=True,
        )
        assert price.price == 0.07
        assert json.loads(route.calls.last.request.content) == {
            "model": "WhisperLargeV3Ct2", "include_ts": True, "ts_level": "word", "include_metadata": True,
            "platform": "tiktok", "duration_seconds": 240,
        }

    @respx.mock
    def test_price_file_multipart(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/audio/transcriptions/price").mock(return_value=httpx.Response(200, json=PRICE))
        client_v2.transcription.create_price(model="m", source_file=b"a", diarize=True)
        req = route.calls.last.request
        assert "multipart/form-data" in req.headers["content-type"]
        assert 'name="diarize"\r\n\r\n1\r\n' in req.content.decode("latin-1")

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_params(self, async_client_v2: AsyncDeapiClient) -> None:
        route = respx.post(f"{V2}/audio/transcriptions").mock(return_value=httpx.Response(200, json=JOB))
        await async_client_v2.transcription.create(
            model="m", source_url="https://x.test/a", ts_level="word", lang="en", include_metadata=True,
        )
        body = json.loads(route.calls.last.request.content)
        assert body["ts_level"] == "word" and body["include_ts"] is True and body["include_metadata"] is True

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_price(self, async_client_v2: AsyncDeapiClient) -> None:
        route = respx.post(f"{V2}/audio/transcriptions/price").mock(return_value=httpx.Response(200, json=PRICE))
        await async_client_v2.transcription.create_price(model="m", duration_seconds=10, diarize=True)
        body = json.loads(route.calls.last.request.content)
        assert body["diarize"] is True and body["include_ts"] is True


# --- get_transcript ---


class TestGetTranscript:
    def test_inline_json_no_http(self, client_v2: DeapiClient) -> None:
        with respx.mock:  # any request would fail: no routes
            t = client_v2.transcription.get_transcript(done(result=json.dumps(CT2_JSON)))
        assert isinstance(t, Transcript) and t.format == "json" and len(t.segments) == 2

    def test_inline_plain_text(self, client_v2: DeapiClient) -> None:
        t = client_v2.transcription.get_transcript(done(result="[0:00 - 0:03] hi"), output_format=None)
        assert t.format == "text" and t.text == "[0:00 - 0:03] hi"

    @respx.mock
    def test_download_has_no_auth_header(self, client_v2: DeapiClient) -> None:
        route = respx.get(STORAGE).mock(return_value=httpx.Response(200, text=json.dumps(CT2_JSON)))
        t = client_v2.transcription.get_transcript(done(result_url=STORAGE), output_format="json")
        assert t.format == "json" and t.language == "en"
        headers = route.calls.last.request.headers
        assert "authorization" not in headers

    @respx.mock
    def test_download_plain_text(self, client_v2: DeapiClient) -> None:
        respx.get(STORAGE).mock(return_value=httpx.Response(200, text="plain words"))
        t = client_v2.transcription.get_transcript(done(result_url=STORAGE))
        assert t.format == "text" and t.text == "plain words"

    @respx.mock
    def test_download_error_raises(self, client_v2: DeapiClient) -> None:
        from deapi import DeapiError

        respx.get(STORAGE).mock(return_value=httpx.Response(404, text="gone"))
        with pytest.raises(DeapiError, match="HTTP 404") as ei:
            client_v2.transcription.get_transcript(done(result_url=STORAGE))
        assert ei.value.status_code == 404

    @respx.mock
    @pytest.mark.parametrize("status", [401, 403])
    def test_storage_auth_error_is_not_an_account_error(self, client_v2: DeapiClient, status: int) -> None:
        from deapi import AuthenticationError, DeapiError, PermissionDeniedError

        respx.get(STORAGE).mock(return_value=httpx.Response(status, text="denied"))
        with pytest.raises(DeapiError) as ei:
            client_v2.transcription.get_transcript(done(result_url=STORAGE))
        assert not isinstance(ei.value, (AuthenticationError, PermissionDeniedError))
        assert ei.value.status_code == status

    @respx.mock
    def test_transport_error_is_network_error(self, client_v2: DeapiClient) -> None:
        from deapi import NetworkError

        respx.get(STORAGE).mock(side_effect=httpx.ConnectError("boom"))
        with pytest.raises(NetworkError):
            client_v2.transcription.get_transcript(done(result_url=STORAGE))

    @respx.mock
    async def test_async_download_error(self, async_client_v2: AsyncDeapiClient) -> None:
        from deapi import DeapiError, NetworkError

        respx.get(STORAGE).mock(return_value=httpx.Response(403))
        with pytest.raises(DeapiError, match="HTTP 403") as ei:
            await async_client_v2.transcription.get_transcript(done(result_url=STORAGE))
        assert type(ei.value) is DeapiError
        respx.get(STORAGE).mock(side_effect=httpx.ConnectError("boom"))
        with pytest.raises(NetworkError):
            await async_client_v2.transcription.get_transcript(done(result_url=STORAGE))

    def test_empty_inline_result_falls_back_to_url(self) -> None:
        from deapi.resources.v2.transcription import _pick_result_source

        assert _pick_result_source(done(result="", result_url=STORAGE), None) == (None, STORAGE)

    def test_create_price_rejects_both_sources(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError, match="not both"):
            client_v2.transcription.create_price(model="m", source_url="https://x.example/a.mp3", source_file=b"x")

    @respx.mock
    def test_alt_format_choice(self, client_v2: DeapiClient) -> None:
        txt_url = "https://storage.example.com/results/tr-1.txt"
        route = respx.get(txt_url).mock(return_value=httpx.Response(200, text='{"text": "looks like json"}'))
        result = done(result=json.dumps(CT2_JSON), result_url=STORAGE, results_alt_formats={"txt": txt_url})
        t = client_v2.transcription.get_transcript(result, alt_format="txt")
        assert route.called
        assert t.format == "text"  # a txt rendition is never decoded
        with pytest.raises(ValueError, match="No 'srt' rendition"):
            client_v2.transcription.get_transcript(result, alt_format="srt")

    def test_nothing_to_read(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError, match="neither"):
            client_v2.transcription.get_transcript(done())

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_inline_and_download(self, async_client_v2: AsyncDeapiClient) -> None:
        t = await async_client_v2.transcription.get_transcript(done(result=json.dumps(CT2_JSON)))
        assert t.format == "json"
        route = respx.get(STORAGE).mock(return_value=httpx.Response(200, text=json.dumps(CT2_JSON)))
        t = await async_client_v2.transcription.get_transcript(done(result_url=STORAGE))
        assert t.format == "json" and t.segments[0].words[0].word == "Hello"
        assert "authorization" not in route.calls.last.request.headers
