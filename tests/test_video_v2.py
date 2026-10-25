"""Tests for the v2 video resource: reference mode, optional card params, webhook secret, descriptions."""

from __future__ import annotations

import json
import re
from urllib.parse import parse_qs

import httpx
import pytest
import respx

from deapi import AsyncDeapiClient, DeapiClient
from deapi.types.common import JobResult
from deapi.types.v1.video import VideoDescription, parse_video_description

V2 = "https://test.deapi.ai/api/v2"
JOB = {"data": {"request_id": "job-1"}}
PRICE = {"data": {"price": 0.5}}
SECRET = "s" * 32

DESCRIPTION = {
    "events": [{"start": 0, "end": 2.5, "description": "A cat jumps", "edge_distance": 0.1}],
    "scenes": [{"start": 0, "end": 10}],
    "meta": {"duration": 10, "preset": "fast", "extra_key": 1},
}


def multipart_fields(request: httpx.Request) -> list[tuple[str, bytes]]:
    """(name, body) of every multipart part, in order."""
    boundary = re.search(r"boundary=([^;]+)", request.headers["content-type"]).group(1).encode()
    parts = []
    for chunk in request.content.split(b"--" + boundary)[1:-1]:
        head, _, body = chunk.partition(b"\r\n\r\n")
        name = re.search(rb'name="([^"]+)"', head).group(1).decode()
        parts.append((name, body[:-2]))
    return parts


def names(request: httpx.Request) -> list[str]:
    return [n for n, _ in multipart_fields(request)]


def form(request: httpx.Request) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(request.content.decode()).items()}


class TestGenerate:
    @respx.mock
    def test_only_set_params_sent(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/generations").mock(return_value=httpx.Response(200, json=JOB))
        client_v2.video.generate(prompt="p", model="m")
        assert json.loads(route.calls.last.request.content) == {"prompt": "p", "model": "m"}

    @respx.mock
    def test_enhance_and_webhook_secret(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/generations").mock(return_value=httpx.Response(200, json=JOB))
        client_v2.video.generate(
            prompt="p", model="m", seed=0, enhance_prompt=False,
            webhook_url="https://x.example/hook", webhook_secret=SECRET,
        )
        body = json.loads(route.calls.last.request.content)
        assert body["enhance_prompt"] is False
        assert body["seed"] == 0
        assert body["webhook_secret"] == SECRET

    def test_secret_without_url_rejected(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError):
            client_v2.video.generate(prompt="p", model="m", webhook_secret=SECRET)

    def test_short_secret_rejected(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError):
            client_v2.video.generate(
                prompt="p", model="m", webhook_url="https://x.example/h", webhook_secret="short"
            )

    @respx.mock
    def test_price_minimal(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/generations/price").mock(return_value=httpx.Response(200, json=PRICE))
        assert client_v2.video.generate_price(prompt="p", model="m", frames=40).price == 0.5
        assert json.loads(route.calls.last.request.content) == {"prompt": "p", "model": "m", "frames": 40}


class TestAnimateReferences:
    @respx.mock
    def test_reference_parts_in_order(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/animations").mock(return_value=httpx.Response(200, json=JOB))
        client_v2.video.animate(
            prompt="<Picture 1> runs", model="m",
            ref_images=[b"img1", b"img2"], ref_videos=[b"vid"], ref_audios=[b"aud"],
            enhance_prompt=True, webhook_url="https://x.example/h", webhook_secret=SECRET,
        )
        fields = multipart_fields(route.calls.last.request)
        file_parts = [(n, b) for n, b in fields if n.endswith("]")]
        assert file_parts == [
            ("ref_images[0]", b"img1"), ("ref_images[1]", b"img2"),
            ("ref_videos[0]", b"vid"), ("ref_audios[0]", b"aud"),
        ]
        by_name = dict(fields)
        assert by_name["enhance_prompt"] == b"1"
        assert by_name["webhook_secret"] == SECRET.encode()
        # card params not set -> not sent
        for absent in ("width", "height", "frames", "fps", "seed", "steps", "guidance"):
            assert absent not in by_name
        assert "first_frame_image" not in by_name

    @respx.mock
    def test_keyframe_mode_still_works(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/animations").mock(return_value=httpx.Response(200, json=JOB))
        client_v2.video.animate(prompt="p", model="m", first_frame_image=b"a", last_frame_image=b"b")
        assert names(route.calls.last.request)[-2:] == ["first_frame_image", "last_frame_image"]

    @pytest.mark.parametrize("keyframe", ["first_frame_image", "last_frame_image"])
    def test_refs_exclusive_with_keyframes(self, client_v2: DeapiClient, keyframe: str) -> None:
        with pytest.raises(ValueError, match="cannot be combined"):
            client_v2.video.animate(prompt="p", model="m", ref_images=[b"x"], **{keyframe: b"y"})

    @pytest.mark.parametrize("field,cap", [("ref_images", 9), ("ref_videos", 1), ("ref_audios", 3)])
    def test_reference_caps(self, client_v2: DeapiClient, field: str, cap: int) -> None:
        kwargs: dict = {"ref_images": [b"i"], field: [b"x"] * (cap + 1)}
        with pytest.raises(ValueError, match=f"{field} accepts at most {cap}"):
            client_v2.video.animate(prompt="p", model="m", **kwargs)

    @respx.mock
    def test_price_audio_count(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/animations/price").mock(return_value=httpx.Response(200, json=PRICE))
        client_v2.video.animate_price(model="m", ref_audios_count=2)
        assert json.loads(route.calls.last.request.content) == {"model": "m", "ref_audios_count": 2}

    def test_audio_alone_rejected(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError, match="ref_audios"):
            client_v2.video.animate(prompt="p", model="m", ref_audios=[b"x"])

    def test_no_source_rejected(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError, match="first_frame_image"):
            client_v2.video.animate(prompt="p", model="m")

    @respx.mock
    def test_video_only_refs_ok(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/animations").mock(return_value=httpx.Response(200, json=JOB))
        client_v2.video.animate(prompt="p", model="m", ref_videos=[b"v"])
        assert "ref_videos[0]" in names(route.calls.last.request)

    @respx.mock
    def test_price_counts(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/animations/price").mock(return_value=httpx.Response(200, json=PRICE))
        client_v2.video.animate_price(model="m", ref_images_count=3, ref_videos_count=1)
        assert json.loads(route.calls.last.request.content) == {
            "model": "m", "ref_images_count": 3, "ref_videos_count": 1,
        }

    @respx.mock
    def test_price_counts_omitted(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/animations/price").mock(return_value=httpx.Response(200, json=PRICE))
        client_v2.video.animate_price(model="m", width=512, height=512, frames=24, fps=24)
        assert "ref_images_count" not in json.loads(route.calls.last.request.content)


class TestAudioAndUpscale:
    @respx.mock
    def test_audio_sync_minimal(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/audio-syncs").mock(return_value=httpx.Response(200, json=JOB))
        client_v2.video.generate_from_audio(
            prompt="p", audio=b"a", model="m", webhook_url="https://x.example/h", webhook_secret=SECRET
        )
        assert sorted(names(route.calls.last.request)) == sorted(
            ["prompt", "model", "webhook_url", "webhook_secret", "audio"]
        )

    @respx.mock
    def test_audio_price_minimal(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/audio-syncs/price").mock(return_value=httpx.Response(200, json=PRICE))
        client_v2.video.generate_from_audio_price(model="m", frames=10)
        assert json.loads(route.calls.last.request.content) == {"model": "m", "frames": 10}

    @respx.mock
    def test_upscale_scale_and_secret(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/upscales").mock(return_value=httpx.Response(200, json=JOB))
        client_v2.video.upscale(
            video=b"v", model="m", scale=3, webhook_url="https://x.example/h", webhook_secret=SECRET
        )
        fields = dict(multipart_fields(route.calls.last.request))
        assert fields["scale"] == b"3"
        assert fields["webhook_secret"] == SECRET.encode()

    @respx.mock
    def test_upscale_without_scale(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/upscales").mock(return_value=httpx.Response(200, json=JOB))
        client_v2.video.upscale(video=b"v", model="m")
        assert "scale" not in names(route.calls.last.request)

    @respx.mock
    def test_upscale_price_by_size(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/upscales/price").mock(return_value=httpx.Response(200, json=PRICE))
        client_v2.video.upscale_price(model="m", width=640, height=360, scale=2, duration=4)
        assert json.loads(route.calls.last.request.content) == {
            "model": "m", "width": 640, "height": 360, "scale": 2, "duration": 4,
        }


class TestDescribe:
    @respx.mock
    def test_url_source(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/descriptions").mock(return_value=httpx.Response(200, json=JOB))
        job = client_v2.video.describe(
            model="Marlin_2B", video_url="https://youtu.be/x", preset="fast",
            include_metadata=True, return_result_in_response=True,
            webhook_url="https://x.example/h", webhook_secret=SECRET,
        )
        assert job.request_id == "job-1"
        assert form(route.calls.last.request) == {
            "model": "Marlin_2B", "video_url": "https://youtu.be/x", "preset": "fast",
            "include_metadata": "1", "return_result_in_response": "1",
            "webhook_url": "https://x.example/h", "webhook_secret": SECRET,
        }

    @respx.mock
    def test_file_source(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/descriptions").mock(return_value=httpx.Response(200, json=JOB))
        client_v2.video.describe(model="Marlin_2B", video_file=b"bytes")
        fields = multipart_fields(route.calls.last.request)
        assert ("video_file", b"bytes") in fields
        assert [n for n, _ in fields] == ["model", "video_file"]

    def test_exactly_one_source(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError):
            client_v2.video.describe(model="m")
        with pytest.raises(ValueError):
            client_v2.video.describe(model="m", video_url="https://a.example/v", video_file=b"x")

    @respx.mock
    def test_metadata_allowed_with_file(self, client_v2: DeapiClient) -> None:
        # Spec: include_metadata is valid for file sources too (returns null metadata, never charged).
        route = respx.post(f"{V2}/videos/descriptions").mock(return_value=httpx.Response(200, json=JOB))
        client_v2.video.describe(model="m", video_file=b"x", include_metadata=True)
        assert dict(multipart_fields(route.calls.last.request))["include_metadata"] == b"1"

    def test_video_param_renamed(self, client_v2: DeapiClient) -> None:
        with pytest.raises(TypeError):
            client_v2.video.describe(model="m", video=b"x")  # type: ignore[call-arg]

    @respx.mock
    def test_price_by_duration(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/descriptions/price").mock(return_value=httpx.Response(200, json=PRICE))
        assert client_v2.video.describe_price(
            model="m", duration_seconds=90, platform="tiktok", preset="balanced"
        ).price == 0.5
        assert form(route.calls.last.request) == {
            "model": "m", "duration_seconds": "90", "platform": "tiktok", "preset": "balanced",
        }

    @respx.mock
    def test_price_by_url_and_file(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/descriptions/price").mock(return_value=httpx.Response(200, json=PRICE))
        client_v2.video.describe_price(model="m", video_url="https://youtu.be/x", include_metadata=False)
        assert form(route.calls.last.request)["include_metadata"] == "0"
        client_v2.video.describe_price(model="m", video_file=b"f")
        assert ("video_file", b"f") in multipart_fields(route.calls.last.request)

    def test_price_source_rules(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError):
            client_v2.video.describe_price(model="m")
        with pytest.raises(ValueError):
            client_v2.video.describe_price(model="m", video_url="https://a.example/v", duration_seconds=5)
        with pytest.raises(ValueError, match="platform"):
            client_v2.video.describe_price(model="m", video_url="https://a.example/v", platform="youtube")


class TestDescriptionResult:
    def test_parse_string_and_dict(self) -> None:
        for raw in (json.dumps(DESCRIPTION), DESCRIPTION):
            parsed = parse_video_description(raw)
            assert isinstance(parsed, VideoDescription)
            assert parsed.events[0].description == "A cat jumps"
            assert parsed.events[0].end == 2.5
            assert parsed.meta["extra_key"] == 1

    def test_parse_rejects_non_object(self) -> None:
        with pytest.raises(ValueError):
            parse_video_description("[1]")

    def test_inline_result(self, client_v2: DeapiClient) -> None:
        result = JobResult(status="done", result=json.dumps(DESCRIPTION))
        assert client_v2.video.get_description(result).scenes == [{"start": 0, "end": 10}]

    @respx.mock
    def test_result_url_download_has_no_auth_header(self, client_v2: DeapiClient) -> None:
        route = respx.get("https://storage.example/out.json").mock(
            return_value=httpx.Response(200, json=DESCRIPTION)
        )
        result = JobResult(status="done", result_url="https://storage.example/out.json")
        assert len(client_v2.video.get_description(result).events) == 1
        assert "authorization" not in route.calls.last.request.headers
        # a bare URL string works as well
        client_v2.video.get_description("https://storage.example/out.json")

    @respx.mock
    def test_download_error(self, client_v2: DeapiClient) -> None:
        respx.get("https://storage.example/gone.json").mock(return_value=httpx.Response(404))
        from deapi._exceptions import DeapiError

        with pytest.raises(DeapiError):
            client_v2.video.get_description("https://storage.example/gone.json")

    @respx.mock
    def test_download_errors_and_bad_json(self, client_v2: DeapiClient) -> None:
        from deapi._exceptions import DeapiError, NetworkError

        url = "https://storage.example/x.json"
        respx.get(url).mock(return_value=httpx.Response(403))
        with pytest.raises(DeapiError, match="HTTP 403") as ei:
            client_v2.video.get_description(url)
        assert type(ei.value) is DeapiError and ei.value.status_code == 403
        respx.get(url).mock(return_value=httpx.Response(200, text="<html>nope"))
        with pytest.raises(DeapiError, match="invalid JSON"):
            client_v2.video.get_description(url)
        respx.get(url).mock(side_effect=httpx.ConnectError("boom"))
        with pytest.raises(NetworkError):
            client_v2.video.get_description(url)

    @respx.mock
    async def test_async_download_errors_and_bad_json(self, async_client_v2: AsyncDeapiClient) -> None:
        from deapi._exceptions import DeapiError, NetworkError

        url = "https://storage.example/x.json"
        respx.get(url).mock(return_value=httpx.Response(500))
        with pytest.raises(DeapiError, match="HTTP 500"):
            await async_client_v2.video.get_description(url)
        respx.get(url).mock(return_value=httpx.Response(200, text="nope"))
        with pytest.raises(DeapiError, match="invalid JSON"):
            await async_client_v2.video.get_description(url)
        respx.get(url).mock(side_effect=httpx.ConnectError("boom"))
        with pytest.raises(NetworkError):
            await async_client_v2.video.get_description(url)

    def test_nothing_to_read(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError):
            client_v2.video.get_description(JobResult(status="processing"))
        with pytest.raises(ValueError):
            client_v2.video.get_description("not-a-url")


class TestLegacyKept:
    @respx.mock
    def test_replace_webhook_secret(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/replacements").mock(return_value=httpx.Response(200, json=JOB))
        client_v2.video.replace(
            video=b"v", ref_image=b"i", model="m", webhook_url="https://x.example/h", webhook_secret=SECRET
        )
        assert dict(multipart_fields(route.calls.last.request))["webhook_secret"] == SECRET.encode()

    @respx.mock
    def test_remove_background_webhook_secret(self, client_v2: DeapiClient) -> None:
        route = respx.post(f"{V2}/videos/background-removals").mock(return_value=httpx.Response(200, json=JOB))
        client_v2.video.remove_background(
            video=b"v", model="m", webhook_url="https://x.example/h", webhook_secret=SECRET
        )
        assert dict(multipart_fields(route.calls.last.request))["webhook_secret"] == SECRET.encode()


class TestAsyncParity:
    @respx.mock
    async def test_generate_and_animate(self, async_client_v2: AsyncDeapiClient) -> None:
        gen = respx.post(f"{V2}/videos/generations").mock(return_value=httpx.Response(200, json=JOB))
        ani = respx.post(f"{V2}/videos/animations").mock(return_value=httpx.Response(200, json=JOB))
        await async_client_v2.video.generate(prompt="p", model="m", enhance_prompt=True)
        assert json.loads(gen.calls.last.request.content) == {"prompt": "p", "model": "m", "enhance_prompt": True}
        await async_client_v2.video.animate(
            prompt="p", model="m", ref_images=[b"1", b"2"], ref_audios=[b"a"], ref_videos=[b"v"]
        )
        assert names(ani.calls.last.request)[-4:] == ["ref_images[0]", "ref_images[1]", "ref_videos[0]", "ref_audios[0]"]
        with pytest.raises(ValueError):
            await async_client_v2.video.animate(prompt="p", model="m", ref_images=[b"1"], first_frame_image=b"f")

    @respx.mock
    async def test_prices(self, async_client_v2: AsyncDeapiClient) -> None:
        route = respx.post(f"{V2}/videos/animations/price").mock(return_value=httpx.Response(200, json=PRICE))
        await async_client_v2.video.animate_price(model="m", ref_images_count=4)
        assert json.loads(route.calls.last.request.content) == {"model": "m", "ref_images_count": 4}
        respx.post(f"{V2}/videos/descriptions/price").mock(return_value=httpx.Response(200, json=PRICE))
        assert (await async_client_v2.video.describe_price(model="m", duration_seconds=30)).price == 0.5

    @respx.mock
    async def test_describe_and_fetch(self, async_client_v2: AsyncDeapiClient) -> None:
        route = respx.post(f"{V2}/videos/descriptions").mock(return_value=httpx.Response(200, json=JOB))
        await async_client_v2.video.describe(model="m", video_file=b"x", webhook_url="https://x.example/h", webhook_secret=SECRET)
        assert dict(multipart_fields(route.calls.last.request))["video_file"] == b"x"
        with pytest.raises(ValueError):
            await async_client_v2.video.describe(model="m")
        dl = respx.get("https://storage.example/o.json").mock(return_value=httpx.Response(200, json=DESCRIPTION))
        out = await async_client_v2.video.get_description(JobResult(status="done", result_url="https://storage.example/o.json"))
        assert out.events[0].start == 0
        assert "authorization" not in dl.calls.last.request.headers
        inline = await async_client_v2.video.get_description(json.dumps(DESCRIPTION))
        assert inline.meta["preset"] == "fast"
