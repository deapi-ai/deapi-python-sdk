from __future__ import annotations

import warnings

import httpx
import pytest
import respx

from deapi import AsyncDeapiClient, DeapiClient
from deapi._types import InferenceType
from deapi.types.common import ModelInfo, ModelsResponse

BASE = "https://test.deapi.ai"
MODELS_URL = f"{BASE}/api/v2/models"


def _model(slug: str, **extra: object) -> dict:
    return {"name": slug, "slug": slug, "inference_types": ["txt2img"], "tags": [], "status": "standard_model",
            "info": {"limits": {"max_side": 1024}, "features": {"supports_steps": True},
                     "defaults": {"width": 512}}, **extra}


def _page(slugs: list[str], page: int, last: int) -> dict:
    return {
        "data": [_model(s) for s in slugs],
        "links": {"first": "f", "last": "l", "prev": None, "next": None},
        "meta": {"current_page": page, "last_page": last, "per_page": 2, "total": 5, "from": 1, "to": 2,
                 "path": MODELS_URL, "links": [{"url": None, "label": "1", "page": 1, "active": True}]},
    }


class TestModelsSync:
    @respx.mock
    def test_limit_and_page_params(self, client_v2: DeapiClient) -> None:
        route = respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=_page(["a"], 2, 3)))
        result = client_v2.models.list(limit=50, page=2)
        params = route.calls.last.request.url.params
        assert params["limit"] == "50"
        assert params["page"] == "2"
        assert "per_page" not in params
        assert result.meta.current_page == 2

    @respx.mock
    def test_default_sends_no_limit(self, client_v2: DeapiClient) -> None:
        route = respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=_page(["a"], 1, 1)))
        client_v2.models.list()
        assert "limit" not in route.calls.last.request.url.params

    @respx.mock
    def test_per_page_deprecated_alias(self, client_v2: DeapiClient) -> None:
        route = respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=_page(["a"], 1, 1)))
        with pytest.warns(DeprecationWarning, match="per_page"):
            client_v2.models.list(per_page=7)
        params = route.calls.last.request.url.params
        assert params["limit"] == "7"
        assert "per_page" not in params

    @respx.mock
    def test_limit_wins_over_per_page(self, client_v2: DeapiClient) -> None:
        route = respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=_page(["a"], 1, 1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            client_v2.models.list(limit=10, per_page=7)
        assert route.calls.last.request.url.params["limit"] == "10"

    @respx.mock
    def test_inference_types_filter(self, client_v2: DeapiClient) -> None:
        route = respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=_page(["a"], 1, 1)))
        client_v2.models.list(inference_types=["txt2img", "img2img"])
        assert route.calls.last.request.url.params["filter[inference_types]"] == "txt2img,img2img"

    @respx.mock
    def test_list_all_walks_pages(self, client_v2: DeapiClient) -> None:
        route = respx.get(MODELS_URL).mock(side_effect=[
            httpx.Response(200, json=_page(["a", "b"], 1, 3)),
            httpx.Response(200, json=_page(["c", "d"], 2, 3)),
            httpx.Response(200, json=_page(["e"], 3, 3)),
        ])
        models = client_v2.models.list_all(inference_types=["txt2img"])
        assert [m.slug for m in models] == ["a", "b", "c", "d", "e"]
        assert [c.request.url.params["page"] for c in route.calls] == ["1", "2", "3"]
        assert all(c.request.url.params["limit"] == "50" for c in route.calls)
        assert all(c.request.url.params["filter[inference_types]"] == "txt2img" for c in route.calls)

    @respx.mock
    def test_list_all_without_meta(self, client_v2: DeapiClient) -> None:
        route = respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json={"data": [_model("a")]}))
        assert [m.slug for m in client_v2.models.list_all()] == ["a"]
        assert route.call_count == 1

    @respx.mock
    def test_get_by_slug(self, client_v2: DeapiClient) -> None:
        respx.get(MODELS_URL).mock(side_effect=[
            httpx.Response(200, json=_page(["a"], 1, 2)),
            httpx.Response(200, json=_page(["b"], 2, 2)),
        ])
        found = client_v2.models.get("b")
        assert found is not None and found.slug == "b"

    @respx.mock
    def test_get_missing_returns_none(self, client_v2: DeapiClient) -> None:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=_page(["a"], 1, 1)))
        assert client_v2.models.get("nope") is None


class TestModelsAsync:
    @respx.mock
    async def test_limit_and_deprecation(self, async_client_v2: AsyncDeapiClient) -> None:
        route = respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=_page(["a"], 1, 1)))
        await async_client_v2.models.list(limit=20, page=3)
        assert route.calls.last.request.url.params["limit"] == "20"
        with pytest.warns(DeprecationWarning):
            await async_client_v2.models.list(per_page=5)
        assert route.calls.last.request.url.params["limit"] == "5"

    @respx.mock
    async def test_list_all_and_get(self, async_client_v2: AsyncDeapiClient) -> None:
        respx.get(MODELS_URL).mock(side_effect=[
            httpx.Response(200, json=_page(["a"], 1, 2)),
            httpx.Response(200, json=_page(["b"], 2, 2)),
            httpx.Response(200, json=_page(["a"], 1, 2)),
            httpx.Response(200, json=_page(["b"], 2, 2)),
        ])
        assert [m.slug for m in await async_client_v2.models.list_all()] == ["a", "b"]
        assert (await async_client_v2.models.get("zzz")) is None


class TestModelTypes:
    def test_lenient_response(self) -> None:
        r = ModelsResponse.model_validate({"data": [_model("a")]})
        assert r.links is None and r.meta is None
        r = ModelsResponse.model_validate({"data": [], "links": [], "meta": {"total": 3, "path": "x"}})
        assert r.links is None
        assert r.meta.total == 3 and r.meta.last_page is None

    def test_live_meta_shape(self) -> None:
        r = ModelsResponse.model_validate(_page(["a"], 1, 1))
        assert r.meta.last_page == 1
        assert r.meta.path == MODELS_URL and r.meta.from_ == 1 and r.meta.to == 2
        assert r.meta.links[0]["label"] == "1"

    def test_model_info_card_fields(self) -> None:
        m = ModelInfo.model_validate(_model("a", tags=["new", "experimental"], status="test_model"))
        assert m.status == "test_model"
        assert m.tags == ["new", "experimental"]
        assert m.limits == {"max_side": 1024}
        assert m.features == {"supports_steps": True}
        assert m.defaults == {"width": 512}

    def test_model_info_empty_info(self) -> None:
        m = ModelInfo.model_validate({"name": "a", "slug": "a", "inference_types": [], "info": [], "tags": None})
        assert m.limits == {} and m.features == {} and m.defaults == {} and m.tags == []

    def test_inference_types_added(self) -> None:
        assert InferenceType.VIDEO2DESC.value == "video2desc"
        assert InferenceType.VIDEO_FILE2DESC.value == "video_file2desc"
        assert InferenceType.AUDIO2VIDEO.value == "audio2video"
