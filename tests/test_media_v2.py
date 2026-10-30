"""v2 behaviour of images, audio, embeddings, OCR and prompt enhancement resources."""

from __future__ import annotations

import json
import warnings

import httpx
import pytest
import respx

from deapi_sdk import AsyncDeapiClient, DeapiClient

V2 = "https://test.deapi.ai/api/v2"
SUBMIT = {"data": {"request_id": "req-1"}}
PRICE = {"data": {"price": 0.5}}
HOOK = "https://example.com/hook"
SECRET = "s" * 32
PNG = b"\x89PNG-bytes"


def _json(route: respx.Route) -> dict:
    return json.loads(route.calls.last.request.content)


def _body(route: respx.Route) -> str:
    return route.calls.last.request.content.decode("utf-8", errors="ignore")


def _is_multipart(route: respx.Route) -> bool:
    return route.calls.last.request.headers["content-type"].startswith("multipart/form-data")


def _mock(path: str, response: dict | None = None) -> respx.Route:
    return respx.post(f"{V2}/{path}").mock(return_value=httpx.Response(200, json=response or SUBMIT))


class TestGenerations:
    @respx.mock
    def test_new_fields_sent(self, client_v2: DeapiClient) -> None:
        route = _mock("images/generations")
        client_v2.images.generate(
            prompt="cat", model="gpt-image-2", quality="high", enhance_prompt=True,
            webhook_url=HOOK, webhook_secret=SECRET,
        )
        body = _json(route)
        assert body["quality"] == "high"
        assert body["enhance_prompt"] is True
        assert body["webhook_url"] == HOOK
        assert body["webhook_secret"] == SECRET

    @respx.mock
    def test_unset_card_fields_omitted(self, client_v2: DeapiClient) -> None:
        route = _mock("images/generations")
        client_v2.images.generate(prompt="cat", model="m")
        assert _json(route) == {"prompt": "cat", "model": "m"}

    @respx.mock
    def test_enhance_false_is_sent(self, client_v2: DeapiClient) -> None:
        route = _mock("images/generations")
        client_v2.images.generate(prompt="cat", model="m", enhance_prompt=False)
        assert _json(route)["enhance_prompt"] is False

    @respx.mock
    def test_price_takes_quality_not_enhance(self, client_v2: DeapiClient) -> None:
        route = _mock("images/generations/price", PRICE)
        result = client_v2.images.generate_price(prompt="cat", model="m", quality="low")
        assert result.price == 0.5
        assert _json(route) == {"prompt": "cat", "model": "m", "quality": "low"}

    def test_secret_without_url(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError, match="webhook_url"):
            client_v2.images.generate(prompt="cat", model="m", webhook_secret=SECRET)

    def test_short_secret(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError, match="32"):
            client_v2.images.generate(prompt="cat", model="m", webhook_url=HOOK, webhook_secret="short")

    @respx.mock
    async def test_async_parity(self, async_client_v2: AsyncDeapiClient) -> None:
        route = _mock("images/generations")
        job = await async_client_v2.images.generate(
            prompt="cat", model="m", quality="medium", enhance_prompt=True,
            webhook_url=HOOK, webhook_secret=SECRET,
        )
        assert job.request_id == "req-1"
        body = _json(route)
        assert body["quality"] == "medium" and body["enhance_prompt"] is True
        assert body["webhook_secret"] == SECRET
        assert "width" not in body and "seed" not in body

    async def test_async_secret_validation(self, async_client_v2: AsyncDeapiClient) -> None:
        with pytest.raises(ValueError):
            await async_client_v2.images.generate(prompt="cat", model="m", webhook_secret=SECRET)


class TestEdits:
    @respx.mock
    def test_single_image_with_new_fields(self, client_v2: DeapiClient) -> None:
        route = _mock("images/edits")
        client_v2.images.transform(
            prompt="x", model="m", image=PNG, quality="high", enhance_prompt=True,
            webhook_url=HOOK, webhook_secret=SECRET,
        )
        body = _body(route)
        assert _is_multipart(route)
        assert 'name="quality"' in body and "high" in body
        assert 'name="enhance_prompt"' in body
        assert 'name="webhook_secret"' in body
        assert 'name="image"' in body
        # steps and seed are no longer required
        assert 'name="steps"' not in body and 'name="seed"' not in body

    @respx.mock
    def test_multiple_images_sent_as_array(self, client_v2: DeapiClient) -> None:
        route = _mock("images/edits")
        client_v2.images.transform(prompt="x", model="m", images=[PNG, PNG, PNG])
        body = _body(route)
        assert body.count('name="images[]"') == 3
        assert 'name="image"' not in body

    def test_image_and_images_exclusive(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError, match="not both"):
            client_v2.images.transform(prompt="x", model="m", image=PNG, images=[PNG])

    def test_source_required(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError, match="required"):
            client_v2.images.transform(prompt="x", model="m")

    @respx.mock
    def test_loras_bracketed(self, client_v2: DeapiClient) -> None:
        route = _mock("images/edits")
        client_v2.images.transform(prompt="x", model="m", image=PNG, loras=[{"name": "a", "weight": 0.5}])
        body = _body(route)
        assert 'name="loras[0][name]"' in body and 'name="loras[0][weight]"' in body

    @respx.mock
    def test_price_json_without_image(self, client_v2: DeapiClient) -> None:
        route = _mock("images/edits/price", PRICE)
        client_v2.images.transform_price(
            prompt="x", model="m", negative_prompt="blur", width=512, height=768, quality="low",
        )
        assert _json(route) == {
            "prompt": "x", "model": "m", "negative_prompt": "blur",
            "width": 512, "height": 768, "quality": "low",
        }

    @respx.mock
    def test_price_multipart_with_images(self, client_v2: DeapiClient) -> None:
        route = _mock("images/edits/price", PRICE)
        client_v2.images.transform_price(prompt="x", model="m", images=[PNG, PNG], quality="high")
        body = _body(route)
        assert _is_multipart(route)
        assert body.count('name="images[]"') == 2
        assert 'name="quality"' in body

    def test_price_exclusive(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError, match="not both"):
            client_v2.images.transform_price(prompt="x", model="m", image=PNG, images=[PNG])

    @respx.mock
    async def test_async_parity(self, async_client_v2: AsyncDeapiClient) -> None:
        route = _mock("images/edits")
        await async_client_v2.images.transform(
            prompt="x", model="m", images=[PNG, PNG], quality="low", enhance_prompt=False,
            webhook_url=HOOK, webhook_secret=SECRET,
        )
        body = _body(route)
        assert body.count('name="images[]"') == 2
        assert 'name="quality"' in body and 'name="webhook_secret"' in body
        with pytest.raises(ValueError, match="not both"):
            await async_client_v2.images.transform(prompt="x", model="m", image=PNG, images=[PNG])

    @respx.mock
    async def test_async_price(self, async_client_v2: AsyncDeapiClient) -> None:
        route = _mock("images/edits/price", PRICE)
        result = await async_client_v2.images.transform_price(prompt="x", model="m", quality="low", width=64)
        assert result.price == 0.5
        assert _json(route)["width"] == 64


class TestUpscaleRemoveBackground:
    @respx.mock
    def test_upscale_webhook_secret(self, client_v2: DeapiClient) -> None:
        route = _mock("images/upscales")
        client_v2.images.upscale(image=PNG, model="m", webhook_url=HOOK, webhook_secret=SECRET)
        assert 'name="webhook_secret"' in _body(route)

    @respx.mock
    def test_rmbg_webhook_secret(self, client_v2: DeapiClient) -> None:
        route = _mock("images/background-removals")
        client_v2.images.remove_background(image=PNG, model="m", webhook_url=HOOK, webhook_secret=SECRET)
        assert 'name="webhook_secret"' in _body(route)

    def test_secret_validation(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError):
            client_v2.images.upscale(image=PNG, model="m", webhook_secret=SECRET)
        with pytest.raises(ValueError):
            client_v2.images.remove_background(image=PNG, model="m", webhook_url="http://x.io", webhook_secret=SECRET)

    @respx.mock
    def test_price_by_size(self, client_v2: DeapiClient) -> None:
        route = _mock("images/upscales/price", PRICE)
        client_v2.images.upscale_price(model="m", width=100, height=200)
        assert _json(route) == {"model": "m", "width": 100, "height": 200}

    @respx.mock
    async def test_async(self, async_client_v2: AsyncDeapiClient) -> None:
        route = _mock("images/background-removals")
        await async_client_v2.images.remove_background(image=PNG, model="m", webhook_url=HOOK, webhook_secret=SECRET)
        assert 'name="webhook_secret"' in _body(route)
        route = _mock("images/upscales")
        await async_client_v2.images.upscale(image=PNG, model="m", webhook_url=HOOK, webhook_secret=SECRET)
        assert 'name="webhook_secret"' in _body(route)


class TestOCR:
    @respx.mock
    def test_extract_webhook_secret(self, client_v2: DeapiClient) -> None:
        route = _mock("images/ocr")
        client_v2.ocr.extract(image=PNG, model="m", webhook_url=HOOK, webhook_secret=SECRET)
        assert 'name="webhook_secret"' in _body(route)

    def test_extract_secret_validation(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError):
            client_v2.ocr.extract(image=PNG, model="m", webhook_secret=SECRET)

    @respx.mock
    def test_price_language(self, client_v2: DeapiClient) -> None:
        route = _mock("images/ocr/price", PRICE)
        client_v2.ocr.extract_price(model="m", width=10, height=20, language="en")
        assert _json(route) == {"model": "m", "language": "en", "width": 10, "height": 20}

    @respx.mock
    async def test_async(self, async_client_v2: AsyncDeapiClient) -> None:
        route = _mock("images/ocr")
        await async_client_v2.ocr.extract(image=PNG, model="m", webhook_url=HOOK, webhook_secret=SECRET)
        assert 'name="webhook_secret"' in _body(route)
        with pytest.raises(ValueError):
            await async_client_v2.ocr.extract(image=PNG, model="m", webhook_secret=SECRET)


class TestEmbeddings:
    @respx.mock
    def test_webhook_secret(self, client_v2: DeapiClient) -> None:
        route = _mock("embeddings")
        client_v2.embeddings.create(input="hi", model="m", webhook_url=HOOK, webhook_secret=SECRET)
        assert _json(route)["webhook_secret"] == SECRET

    def test_secret_validation(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError):
            client_v2.embeddings.create(input="hi", model="m", webhook_secret=SECRET)

    @respx.mock
    async def test_async(self, async_client_v2: AsyncDeapiClient) -> None:
        route = _mock("embeddings")
        await async_client_v2.embeddings.create(input=["a"], model="m", webhook_url=HOOK, webhook_secret=SECRET)
        assert _json(route)["webhook_secret"] == SECRET
        with pytest.raises(ValueError):
            await async_client_v2.embeddings.create(input="hi", model="m", webhook_secret=SECRET)


class TestSpeech:
    @respx.mock
    def test_card_defaults_omitted(self, client_v2: DeapiClient) -> None:
        route = _mock("audio/speech")
        client_v2.audio.synthesize(text="hello there", model="Kokoro")
        assert _json(route) == {"text": "hello there", "model": "Kokoro"}

    @respx.mock
    def test_explicit_values_and_webhook(self, client_v2: DeapiClient) -> None:
        route = _mock("audio/speech")
        client_v2.audio.synthesize(
            text="hello there", model="Kokoro", lang="en-us", format="mp3", speed=1.0,
            sample_rate=24000, voice="af_alloy", webhook_url=HOOK, webhook_secret=SECRET,
        )
        body = _json(route)
        assert body["lang"] == "en-us" and body["sample_rate"] == 24000
        assert body["webhook_secret"] == SECRET

    @respx.mock
    def test_ref_audio_multipart(self, client_v2: DeapiClient) -> None:
        route = _mock("audio/speech")
        client_v2.audio.synthesize(
            text="hello there", model="m", mode="voice_clone", ref_audio=b"RIFF", ref_text="hi",
            webhook_url=HOOK, webhook_secret=SECRET,
        )
        body = _body(route)
        assert _is_multipart(route)
        assert 'name="ref_audio"' in body and 'name="webhook_secret"' in body
        assert 'name="lang"' not in body

    @respx.mock
    def test_price_minimal(self, client_v2: DeapiClient) -> None:
        route = _mock("audio/speech/price", PRICE)
        client_v2.audio.synthesize_price(model="Kokoro", count_text=100)
        assert _json(route) == {"model": "Kokoro", "count_text": 100}

    def test_secret_validation(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError):
            client_v2.audio.synthesize(text="hello there", model="m", webhook_secret=SECRET)

    @respx.mock
    async def test_async(self, async_client_v2: AsyncDeapiClient) -> None:
        route = _mock("audio/speech")
        await async_client_v2.audio.synthesize(text="hello there", model="m", webhook_url=HOOK, webhook_secret=SECRET)
        body = _json(route)
        assert body == {"text": "hello there", "model": "m", "webhook_url": HOOK, "webhook_secret": SECRET}
        route = _mock("audio/speech/price", PRICE)
        await async_client_v2.audio.synthesize_price(model="m", text="hello")
        assert _json(route) == {"model": "m", "text": "hello"}


class TestMusic:
    @respx.mock
    def test_json_without_reference(self, client_v2: DeapiClient) -> None:
        route = _mock("audio/music")
        client_v2.audio.compose(caption="lofi", model="AceStep_1_5_Turbo", duration=30, lyrics="[Instrumental]")
        assert not _is_multipart(route)
        assert _json(route) == {
            "caption": "lofi", "model": "AceStep_1_5_Turbo", "duration": 30, "lyrics": "[Instrumental]",
        }

    @respx.mock
    def test_reference_audio_multipart(self, client_v2: DeapiClient) -> None:
        route = _mock("audio/music")
        client_v2.audio.compose(
            caption="lofi", model="m", duration=30, lyrics="[Instrumental]", reference_audio=b"ID3",
            seed=7, bpm=90, timesignature=4, webhook_url=HOOK, webhook_secret=SECRET,
        )
        body = _body(route)
        assert _is_multipart(route)
        assert 'name="reference_audio"' in body
        assert 'name="seed"' in body and 'name="timesignature"' in body
        assert 'name="webhook_secret"' in body
        assert 'name="inference_steps"' not in body and 'name="format"' not in body

    @respx.mock
    def test_explicit_card_params_sent(self, client_v2: DeapiClient) -> None:
        route = _mock("audio/music")
        client_v2.audio.compose(
            caption="c", model="m", duration=10, inference_steps=8, guidance_scale=1.0, seed=-1, format="mp3",
        )
        body = _json(route)
        assert body["inference_steps"] == 8 and body["guidance_scale"] == 1.0
        assert body["seed"] == -1 and body["format"] == "mp3"

    def test_secret_validation(self, client_v2: DeapiClient) -> None:
        with pytest.raises(ValueError):
            client_v2.audio.compose(caption="c", model="m", duration=10, webhook_secret=SECRET)
        with pytest.raises(ValueError):
            client_v2.audio.compose(
                caption="c", model="m", duration=10, reference_audio=b"ID3", webhook_url=HOOK, webhook_secret="x",
            )

    @respx.mock
    def test_price_inference_steps_optional(self, client_v2: DeapiClient) -> None:
        route = _mock("audio/music/price", PRICE)
        client_v2.audio.compose_price(model="m", duration=30)
        assert _json(route) == {"model": "m", "duration": 30}
        client_v2.audio.compose_price(model="m", duration=30, inference_steps=8)
        assert _json(route)["inference_steps"] == 8

    @respx.mock
    async def test_async_reference_audio(self, async_client_v2: AsyncDeapiClient) -> None:
        route = _mock("audio/music")
        await async_client_v2.audio.compose(
            caption="c", model="m", duration=10, reference_audio=b"ID3", webhook_url=HOOK, webhook_secret=SECRET,
        )
        assert _is_multipart(route)
        assert 'name="reference_audio"' in _body(route)
        route = _mock("audio/music")
        await async_client_v2.audio.compose(caption="c", model="m", duration=10)
        assert not _is_multipart(route)
        with pytest.raises(ValueError):
            await async_client_v2.audio.compose(caption="c", model="m", duration=10, webhook_secret=SECRET)


class TestPrompts:
    @respx.mock
    def test_enhance_sends_type_and_model_slug(self, client_v2: DeapiClient) -> None:
        route = _mock("prompts/enhancements", {"prompt": "better", "negative_prompt": None})
        result = client_v2.prompts.enhance(prompt="a cat", type="audio.music", model_slug="AceStep_1_5_Turbo")
        assert result.prompt == "better"
        body = _body(route)
        assert 'name="type"' in body and "audio.music" in body
        assert 'name="model_slug"' in body

    @respx.mock
    def test_price_top_level_shape(self, client_v2: DeapiClient) -> None:
        _mock("prompts/enhancements/price", {"price": 0.01})
        result = client_v2.prompts.enhance_price(prompt="a cat", type="images.edits", model_slug="m", image=PNG)
        assert result.price == 0.01

    @respx.mock
    def test_legacy_video_price_routes_animations_with_image(self, client_v2: DeapiClient) -> None:
        route = _mock("prompts/enhancements/price", {"price": 0.02})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            client_v2.prompts.enhance_video_price(prompt="a cat", model_slug="m", image=PNG)
        body = _body(route)
        assert "videos.animations" in body and 'name="image"' in body

    @respx.mock
    def test_legacy_video_price_without_image(self, client_v2: DeapiClient) -> None:
        route = _mock("prompts/enhancements/price", {"price": 0.02})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            client_v2.prompts.enhance_video_price(prompt="a cat", model_slug="m")
        assert "videos.generations" in _body(route)

    @respx.mock
    async def test_async_legacy_video_price(self, async_client_v2: AsyncDeapiClient) -> None:
        route = _mock("prompts/enhancements/price", {"price": 0.02})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            await async_client_v2.prompts.enhance_video_price(prompt="a cat", model_slug="m", image=PNG)
        assert "videos.animations" in _body(route)

    def test_booster_types_exported(self) -> None:
        from typing import get_args

        from deapi_sdk.types.v1.prompts import PromptBoosterType

        assert set(get_args(PromptBoosterType)) == {
            "images.generations", "images.edits", "videos.generations",
            "videos.animations", "audio.speech", "audio.music",
        }
