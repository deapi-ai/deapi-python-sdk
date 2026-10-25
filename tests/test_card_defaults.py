"""Unset card-driven parameters are filled from the model card (``info.defaults``)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from deapi import AsyncDeapiClient, DeapiClient
from deapi._cards import fill_from_card

BASE = "https://test.deapi.ai"
V2 = f"{BASE}/api/v2"
KEY = "test-key-123"
SUBMIT = {"data": {"request_id": "req-1"}}
PRICE = {"data": {"price": 0.5}}
PNG = b"\x89PNG-bytes"

FLUX = {
    "slug": "Flux1schnell",
    "info": {"defaults": {"steps": 4, "width": 768, "height": 768, "quality": None,
                          "negative_prompt": "Negative prompt"}},
}
VIDEO = {"slug": "Ltx", "info": {"defaults": {"width": 512, "height": 512, "frames": 97, "fps": 24,
                                              "steps": 20, "prompt": "placeholder"}}}
TTS = {"slug": "Chatterbox", "info": {"defaults": {"lang": "en", "speed": 1, "voice": "default",
                                                   "format": "mp3", "sample_rate": 24000}}}
MUSIC = {"slug": "Ace", "info": {"defaults": {"format": "mp3"}}}
PAGE1 = [FLUX, VIDEO]
PAGE2 = [TTS, MUSIC]


def _models(pages: tuple[list[dict[str, Any]], ...] = (PAGE1, PAGE2)) -> respx.Route:
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params.get("page", "1"))
        return httpx.Response(
            200, json={"data": pages[page - 1], "meta": {"current_page": page, "last_page": len(pages)}}
        )

    return respx.get(f"{V2}/models").mock(side_effect=handler)


def _post(path: str, response: dict | None = None) -> respx.Route:
    return respx.post(f"{V2}/{path}").mock(return_value=httpx.Response(200, json=response or SUBMIT))


def _json(route: respx.Route) -> dict:
    return json.loads(route.calls.last.request.content)


def _form(route: respx.Route) -> str:
    return route.calls.last.request.content.decode("utf-8", errors="ignore")


def _client(**kw: Any) -> DeapiClient:
    return DeapiClient(api_key=KEY, base_url=BASE, api_version="v2", **kw)


def test_fill_from_card_helper() -> None:
    out = fill_from_card({"width": None, "height": 5}, {"width": 1, "height": 2, "steps": None}, ["width", "height", "steps"])
    assert out == {"width": 1, "height": 5}


class TestImages:
    @respx.mock
    def test_job_and_price_filled(self) -> None:
        models = _models()
        gen = _post("images/generations")
        price = _post("images/generations/price", PRICE)
        with _client() as c:
            c.images.generate(prompt="cat", model="Flux1schnell")
            c.images.generate_price(prompt="cat", model="Flux1schnell")
        for route in (gen, price):
            assert _json(route) == {
                "prompt": "cat", "model": "Flux1schnell", "width": 768, "height": 768,
                "steps": 4, "seed": -1,
            }
        assert models.call_count == 2  # two pages, fetched once for both calls

    @respx.mock
    def test_user_value_wins_and_free_text_not_filled(self) -> None:
        _models()
        gen = _post("images/generations")
        with _client() as c:
            c.images.generate(prompt="cat", model="Flux1schnell", width=1024, seed=7)
        body = _json(gen)
        assert body["width"] == 1024 and body["seed"] == 7 and body["height"] == 768
        assert "negative_prompt" not in body and "quality" not in body

    @respx.mock
    def test_multipart_values_are_strings(self) -> None:
        _models()
        route = _post("images/edits")
        with _client() as c:
            c.images.transform(prompt="x", model="Flux1schnell", image=PNG)
        text = _form(route)
        for name, value in (("width", "768"), ("height", "768"), ("steps", "4"), ("seed", "-1")):
            assert f'name="{name}"\r\n\r\n{value}\r\n' in text

    @respx.mock
    def test_transform_price_json_without_image(self) -> None:
        _models()
        route = _post("images/edits/price", PRICE)
        with _client() as c:
            c.images.transform_price(prompt="x", model="Flux1schnell")
        assert _json(route)["seed"] == -1 and _json(route)["width"] == 768


class TestVideoAudio:
    @respx.mock
    def test_video_generate_and_price(self) -> None:
        _models()
        gen = _post("videos/generations")
        price = _post("videos/generations/price", PRICE)
        with _client() as c:
            c.video.generate(prompt="p", model="Ltx")
            c.video.generate_price(prompt="p", model="Ltx")
        assert _json(gen) == {"prompt": "p", "model": "Ltx", "width": 512, "height": 512,
                              "frames": 97, "fps": 24, "seed": -1}
        assert "seed" not in _json(price) and _json(price)["fps"] == 24

    @respx.mock
    def test_animate_multipart_strings(self) -> None:
        _models()
        route = _post("videos/animations")
        with _client() as c:
            c.video.animate(prompt="p", model="Ltx", first_frame_image=PNG, fps=30)
        text = _form(route)
        assert 'name="frames"\r\n\r\n97\r\n' in text
        assert 'name="fps"\r\n\r\n30\r\n' in text
        assert 'name="seed"\r\n\r\n-1\r\n' in text

    @respx.mock
    def test_audio_to_video_price(self) -> None:
        _models()
        route = _post("videos/audio-syncs/price", PRICE)
        with _client() as c:
            c.video.generate_from_audio_price(model="Ltx")
        assert _json(route)["frames"] == 97

    @respx.mock
    def test_tts_json_and_price(self) -> None:
        _models()
        gen = _post("audio/speech")
        price = _post("audio/speech/price", PRICE)
        with _client() as c:
            c.audio.synthesize(text="hi", model="Chatterbox", speed=1.5)
            c.audio.synthesize_price(model="Chatterbox", text="hi")
        body = _json(gen)
        assert body == {"text": "hi", "model": "Chatterbox", "lang": "en", "speed": 1.5,
                        "format": "mp3", "sample_rate": 24000}
        assert _json(price)["sample_rate"] == 24000 and "voice" not in _json(price)

    @respx.mock
    def test_music_seed_and_format(self) -> None:
        _models()
        gen = _post("audio/music")
        price = _post("audio/music/price", PRICE)
        with _client() as c:
            c.audio.compose(caption="c", model="Ace", duration=30, lyrics="[Instrumental]")
            c.audio.compose_price(model="Ace", duration=30)
        body = _json(gen)
        assert body["format"] == "mp3" and body["seed"] == -1
        assert "inference_steps" not in body and "lyrics" in body
        assert "inference_steps" not in _json(price)  # card has no default: left for the API to report


class TestFallbacksAndOptOut:
    @respx.mock
    def test_unknown_model_fills_nothing(self) -> None:
        _models()
        gen = _post("images/generations")
        with _client() as c:
            c.images.generate(prompt="cat", model="Nope")
        assert _json(gen) == {"prompt": "cat", "model": "Nope"}

    @pytest.mark.parametrize("status", [401, 403, 500])
    @respx.mock
    def test_fetch_failure_fills_nothing_and_is_remembered(self, status: int) -> None:
        models = respx.get(f"{V2}/models").mock(return_value=httpx.Response(status, json={"message": "x"}))
        gen = _post("images/generations")
        with _client(max_retries=0) as c:
            c.images.generate(prompt="cat", model="Flux1schnell")
            c.images.generate(prompt="dog", model="Flux1schnell")
        assert _json(gen) == {"prompt": "dog", "model": "Flux1schnell"}
        assert models.call_count == 1

    @respx.mock
    def test_unexpected_shape_fills_nothing(self) -> None:
        respx.get(f"{V2}/models").mock(return_value=httpx.Response(200, json={"oops": 1}))
        gen = _post("images/generations")
        with _client() as c:
            c.images.generate(prompt="cat", model="Flux1schnell")
        assert _json(gen) == {"prompt": "cat", "model": "Flux1schnell"}

    @respx.mock
    def test_flag_disables(self) -> None:
        models = _models()
        gen = _post("images/generations")
        with _client(fill_model_defaults=False) as c:
            c.images.generate(prompt="cat", model="Flux1schnell")
        assert _json(gen) == {"prompt": "cat", "model": "Flux1schnell"}
        assert models.call_count == 0

    @pytest.mark.parametrize("value", ["0", "false", "False"])
    @respx.mock
    def test_env_var_disables(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("DEAPI_FILL_MODEL_DEFAULTS", value)
        models = _models()
        gen = _post("images/generations")
        with _client() as c:
            c.images.generate(prompt="cat", model="Flux1schnell")
        assert _json(gen) == {"prompt": "cat", "model": "Flux1schnell"}
        assert models.call_count == 0

    @respx.mock
    def test_explicit_flag_beats_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEAPI_FILL_MODEL_DEFAULTS", "0")
        _models()
        gen = _post("images/generations")
        with _client(fill_model_defaults=True) as c:
            c.images.generate(prompt="cat", model="Flux1schnell")
        assert _json(gen)["width"] == 768


class TestAsyncParity:
    @respx.mock
    async def test_async_fill_once_per_client(self) -> None:
        models = _models()
        gen = _post("images/generations")
        edits = _post("images/edits")
        speech = _post("audio/speech")
        async with AsyncDeapiClient(api_key=KEY, base_url=BASE, api_version="v2") as c:
            await c.images.generate(prompt="cat", model="Flux1schnell")
            await c.images.transform(prompt="x", model="Flux1schnell", image=PNG)
            await c.audio.synthesize(text="hi", model="Chatterbox")
        assert _json(gen)["seed"] == -1 and _json(gen)["steps"] == 4
        assert 'name="width"\r\n\r\n768\r\n' in _form(edits)
        assert _json(speech)["lang"] == "en"
        assert models.call_count == 2

    @respx.mock
    async def test_async_failure_and_opt_out(self) -> None:
        respx.get(f"{V2}/models").mock(return_value=httpx.Response(403, json={"message": "no"}))
        gen = _post("images/generations")
        async with AsyncDeapiClient(api_key=KEY, base_url=BASE, api_version="v2", max_retries=0) as c:
            await c.images.generate(prompt="cat", model="Flux1schnell")
        assert _json(gen) == {"prompt": "cat", "model": "Flux1schnell"}

    @respx.mock
    async def test_async_opt_out(self) -> None:
        models = _models()
        _post("images/generations/price", PRICE)
        async with AsyncDeapiClient(
            api_key=KEY, base_url=BASE, api_version="v2", fill_model_defaults=False
        ) as c:
            await c.images.generate_price(prompt="cat", model="Flux1schnell")
        assert models.call_count == 0
