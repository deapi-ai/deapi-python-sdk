"""Integration tests that run against the production DeAPI server.

These tests are SKIPPED by default. To run them:

    export DEAPI_API_KEY="sk-your-real-key"
    pytest tests/test_integration.py -m integration -v

You can also run all tests including integration:

    pytest -m integration -v

These tests will incur real API costs. Use a test account or budget accordingly.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from deapi import AsyncDeapiClient, DeapiClient
from deapi._exceptions import AuthenticationError, ValidationError
from deapi._types import JobStatus
from deapi.types.common import ModelInfo

# Skip all tests in this module unless DEAPI_API_KEY is set and -m integration is used
pytestmark = pytest.mark.integration

DEAPI_API_KEY = os.environ.get("DEAPI_API_KEY", "")

requires_api_key = pytest.mark.skipif(
    not DEAPI_API_KEY,
    reason="DEAPI_API_KEY environment variable not set",
)


@pytest.fixture
def live_client() -> DeapiClient:
    """Create a real client pointed at production."""
    return DeapiClient(api_key=DEAPI_API_KEY)


@pytest.fixture
async def live_async_client() -> AsyncDeapiClient:
    """Create a real async client pointed at production."""
    return AsyncDeapiClient(api_key=DEAPI_API_KEY)


# ---------------------------------------------------------------------------
# Helpers — dynamically discover models from the API
# ---------------------------------------------------------------------------


def _find_model(client: DeapiClient, inference_type: str) -> ModelInfo | None:
    """Find the first active model supporting a given inference type."""
    models = client.models.list(inference_types=[inference_type], per_page=5)
    return models.data[0] if models.data else None


def _build_txt2img_params(model: ModelInfo) -> dict[str, Any]:
    """Build valid generation params from a model's info (limits/defaults/features)."""
    info = model.info or {}
    defaults = info.get("defaults", {})
    limits = info.get("limits", {})
    features = info.get("features", {})

    params: dict[str, Any] = {
        "prompt": "a simple test image",
        "model": model.slug,
        "width": int(defaults.get("width", limits.get("min_width", 512))),
        "height": int(defaults.get("height", limits.get("min_height", 512))),
        "seed": 42,
    }
    # Add steps if model has them
    if "steps" in defaults:
        params["steps"] = int(defaults["steps"])
    elif "min_steps" in limits:
        params["steps"] = int(limits["min_steps"])
    # Add guidance only if model supports it
    if features.get("supports_guidance"):
        params["guidance"] = float(defaults.get("guidance", 7.5))
    return params


# ---------------------------------------------------------------------------
# Account / Auth
# ---------------------------------------------------------------------------


class TestAccountIntegration:
    @requires_api_key
    def test_balance_returns_valid_response(self, live_client: DeapiClient) -> None:
        balance = live_client.balance()
        assert balance.balance >= 0

    @requires_api_key
    def test_invalid_api_key_raises_auth_error(self) -> None:
        bad_client = DeapiClient(api_key="sk-invalid-key-123")
        with pytest.raises(AuthenticationError):
            bad_client.balance()

    @requires_api_key
    def test_models_list(self, live_client: DeapiClient) -> None:
        models = live_client.models.list(per_page=5)
        assert len(models.data) > 0
        assert models.meta.current_page == 1
        for model in models.data:
            assert model.name
            assert model.slug
            assert isinstance(model.inference_types, list)

    @requires_api_key
    def test_models_filter_by_inference_type(self, live_client: DeapiClient) -> None:
        models = live_client.models.list(inference_types=["txt2img"], per_page=5)
        assert len(models.data) > 0
        for model in models.data:
            assert "txt2img" in model.inference_types

    @requires_api_key
    def test_model_has_info(self, live_client: DeapiClient) -> None:
        """Models should include info with at least limits."""
        models = live_client.models.list(per_page=5)
        models_with_info = [m for m in models.data if m.info]
        assert len(models_with_info) > 0
        for model in models_with_info:
            assert "limits" in model.info


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


class TestImagesIntegration:
    @requires_api_key
    def test_generate_price(self, live_client: DeapiClient) -> None:
        model = _find_model(live_client, "txt2img")
        assert model, "No txt2img model available"
        params = _build_txt2img_params(model)
        price = live_client.images.generate_price(**params)
        assert price.price > 0

    @requires_api_key
    def test_generate_and_wait(self, live_client: DeapiClient) -> None:
        """Full round trip: submit, poll, get result."""
        model = _find_model(live_client, "txt2img")
        assert model, "No txt2img model available"
        params = _build_txt2img_params(model)
        job = live_client.images.generate(**params)
        assert job.request_id

        result = job.wait(max_wait=120.0)
        assert result.status == JobStatus.DONE
        assert result.result_url is not None
        assert result.result_url.startswith("http")

    @requires_api_key
    def test_generate_validation_error(self, live_client: DeapiClient) -> None:
        """Invalid model name should raise ValidationError."""
        with pytest.raises(ValidationError):
            live_client.images.generate(
                prompt="test",
                model="nonexistent-model-xyz-999",
                width=512,
                height=512,
                seed=42,
            )


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


class TestTranscriptionIntegration:
    @requires_api_key
    def test_transcription_price_by_duration(self, live_client: DeapiClient) -> None:
        """Use legacy vid2txt price endpoint with duration_seconds."""
        model = _find_model(live_client, "video2text")
        assert model, "No video2text model available"
        price = live_client.transcription.from_video_url_price(
            include_ts=True,
            model=model.slug,
            duration_seconds=60.0,
        )
        assert price.price > 0


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class TestEmbeddingsIntegration:
    @requires_api_key
    def test_embeddings_price(self, live_client: DeapiClient) -> None:
        model = _find_model(live_client, "txt2embedding")
        assert model, "No txt2embedding model available"
        price = live_client.embeddings.create_price(
            input="Hello, world!",
            model=model.slug,
        )
        assert price.price >= 0

    @requires_api_key
    def test_embeddings_create_inline(self, live_client: DeapiClient) -> None:
        """Generate embeddings with inline result."""
        model = _find_model(live_client, "txt2embedding")
        assert model, "No txt2embedding model available"
        job = live_client.embeddings.create(
            input="The quick brown fox jumps over the lazy dog",
            model=model.slug,
            return_result_in_response=True,
        )
        assert job.request_id
        result = job.wait(max_wait=60.0)
        assert result.status == JobStatus.DONE


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


class TestPromptsIntegration:
    @requires_api_key
    def test_enhance_image_prompt(self, live_client: DeapiClient) -> None:
        result = live_client.prompts.enhance_image(prompt="cat in space")
        assert result.prompt
        assert len(result.prompt) > len("cat in space")

    @requires_api_key
    def test_enhance_image_price(self, live_client: DeapiClient) -> None:
        price = live_client.prompts.enhance_image_price(prompt="cat in space")
        assert price.price >= 0

    @requires_api_key
    def test_enhance_speech_prompt(self, live_client: DeapiClient) -> None:
        result = live_client.prompts.enhance_speech(prompt="hello world")
        assert result.prompt

    @requires_api_key
    def test_sample_prompt(self, live_client: DeapiClient) -> None:
        result = live_client.prompts.samples(type="text2image")
        assert result.prompt
        assert result.type


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


class TestAsyncIntegration:
    @requires_api_key
    async def test_async_balance(self, live_async_client: AsyncDeapiClient) -> None:
        balance = await live_async_client.balance()
        assert balance.balance >= 0

    @requires_api_key
    async def test_async_models_list(self, live_async_client: AsyncDeapiClient) -> None:
        models = await live_async_client.models.list(per_page=5)
        assert len(models.data) > 0

    @requires_api_key
    async def test_async_generate_price(self, live_async_client: AsyncDeapiClient) -> None:
        """Discover a txt2img model and check its price."""
        models = await live_async_client.models.list(
            inference_types=["txt2img"], per_page=1,
        )
        assert len(models.data) > 0, "No txt2img model available"
        model = models.data[0]
        params = _build_txt2img_params(model)
        params.pop("prompt")
        price = await live_async_client.images.generate_price(prompt="test", **params)
        assert price.price > 0
