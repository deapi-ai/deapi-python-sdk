from __future__ import annotations

import httpx
import pytest
import respx

from deapi_sdk import DeapiClient
from deapi_sdk._exceptions import JobTimeoutError
from deapi_sdk._types import JobStatus

TEST_BASE_URL = "https://test.deapi.ai"
TEST_API_KEY = "test-key-123"
STATUS_URL = f"{TEST_BASE_URL}/api/v2/jobs"

SUBMIT_RESPONSE = {"data": {"request_id": "job-123"}}


class TestPolling:
    @respx.mock
    def test_job_status_single_poll(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, fill_model_defaults=False)
        respx.post(f"{TEST_BASE_URL}/api/v2/images/generations").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        respx.get(f"{STATUS_URL}/job-123").mock(
            return_value=httpx.Response(200, json={
                "data": {
                    "status": "processing",
                    "preview": "https://preview.url",
                    "result_url": None,
                    "result": None,
                    "progress": 45.5,
                }
            })
        )
        job = client.images.generate(prompt="test", model="sdxl", width=512, height=512, seed=1)
        result = job.status()
        assert result.status == JobStatus.PROCESSING
        assert result.progress == 45.5
        assert result.preview == "https://preview.url"

    @respx.mock
    def test_job_wait_polls_until_done(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, fill_model_defaults=False)
        respx.post(f"{TEST_BASE_URL}/api/v2/images/generations").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        status_route = respx.get(f"{STATUS_URL}/job-123")
        status_route.side_effect = [
            httpx.Response(200, json={"data": {"status": "pending", "progress": 0.0}}),
            httpx.Response(200, json={"data": {"status": "processing", "progress": 50.0}}),
            httpx.Response(200, json={
                "data": {
                    "status": "done",
                    "progress": 100.0,
                    "result_url": "https://result.url/image.png",
                    "results_alt_formats": {"jpg": "https://result.url/image.jpg", "webp": "https://result.url/image.webp"},
                }
            }),
        ]
        job = client.images.generate(prompt="test", model="sdxl", width=512, height=512, seed=1)
        result = job.wait(poll_interval=0.01, max_wait=5.0)
        assert result.status == JobStatus.DONE
        assert result.result_url == "https://result.url/image.png"
        assert result.results_alt_formats is not None
        assert status_route.call_count == 3

    @respx.mock
    def test_job_wait_returns_on_error(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, fill_model_defaults=False)
        respx.post(f"{TEST_BASE_URL}/api/v2/images/generations").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        respx.get(f"{STATUS_URL}/job-123").mock(
            return_value=httpx.Response(200, json={"data": {"status": "error", "progress": 0.0}})
        )
        job = client.images.generate(prompt="test", model="sdxl", width=512, height=512, seed=1)
        result = job.wait(poll_interval=0.01)
        assert result.status == JobStatus.ERROR

    @respx.mock
    def test_job_wait_timeout(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, fill_model_defaults=False)
        respx.post(f"{TEST_BASE_URL}/api/v2/images/generations").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        respx.get(f"{STATUS_URL}/job-123").mock(
            return_value=httpx.Response(200, json={"data": {"status": "pending", "progress": 0.0}})
        )
        job = client.images.generate(prompt="test", model="sdxl", width=512, height=512, seed=1)
        with pytest.raises(JobTimeoutError, match="did not complete"):
            job.wait(poll_interval=0.01, max_wait=0.05)

    @respx.mock
    def test_job_repr(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, fill_model_defaults=False)
        respx.post(f"{TEST_BASE_URL}/api/v2/images/generations").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.images.generate(prompt="test", model="sdxl", width=512, height=512, seed=1)
        assert repr(job) == "Job(request_id='job-123')"


class TestModels:
    @respx.mock
    def test_list_models(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, fill_model_defaults=False)
        respx.get(f"{TEST_BASE_URL}/api/v2/models").mock(
            return_value=httpx.Response(200, json={
                "data": [
                    {"name": "sdxl", "slug": "sdxl", "inference_types": ["txt2img", "img2img"], "info": {}},
                    {"name": "whisper", "slug": "whisper", "inference_types": ["audio2text"], "info": {}},
                ],
                "links": {"first": "...", "last": "...", "prev": None, "next": None},
                "meta": {"current_page": 1, "last_page": 1, "per_page": 15, "total": 2},
            })
        )
        result = client.models.list()
        assert len(result.data) == 2
        assert result.meta.total == 2

    @respx.mock
    def test_list_models_with_filter(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, fill_model_defaults=False)
        route = respx.get(f"{TEST_BASE_URL}/api/v2/models").mock(
            return_value=httpx.Response(200, json={
                "data": [{"name": "sdxl", "slug": "sdxl", "inference_types": ["txt2img"], "info": {}}],
                "links": {"first": "...", "last": "...", "prev": None, "next": None},
                "meta": {"current_page": 1, "last_page": 1, "per_page": 15, "total": 1},
            })
        )
        client.models.list(inference_types=["txt2img", "img2img"])
        request = route.calls.last.request
        assert "filter%5Binference_types%5D=txt2img%2Cimg2img" in str(request.url)




class TestJobResultFields:
    def test_new_fields_and_extra(self) -> None:
        from deapi_sdk.types.common import JobResult

        r = JobResult.model_validate({
            "status": "error", "progress": None, "error_code": "PROCESSING_ERROR", "error_reason": "boom",
            "retryable": True, "refunded": True, "failed_at": "2026-01-01T00:00:00Z",
            "price": {"amount": 0.02, "is_estimated": False}, "metadata": {"a": 1}, "prompt_boosted": True,
            "prompt_boost": {"prompt": "p2", "prompt_original": "p", "negative_prompt": None},
            "reference_labels": ["a"], "brand_new": 7,
        })
        assert r.progress is None
        assert r.error_code == "PROCESSING_ERROR" and r.retryable is True and r.refunded is True
        assert r.price["amount"] == 0.02
        assert r.prompt_boost.prompt_original == "p"
        assert r.brand_new == 7

    def test_minimal_still_works(self) -> None:
        from deapi_sdk.types.common import JobResult

        r = JobResult.model_validate({"status": "pending"})
        assert r.progress == 0.0 and r.error_code is None and r.prompt_boost is None

    def test_price_and_balance_extra(self) -> None:
        from deapi_sdk.types.common import Balance, PriceResult

        p = PriceResult.model_validate({"price": 0.1, "is_estimated": True, "estimate_basis": "model_history", "x": 1})
        assert p.is_estimated is True and p.estimate_basis == "model_history" and p.x == 1
        assert PriceResult.model_validate({"price": 0.1}).is_estimated is None
        # The spec requires nothing: an unavailable estimate may carry no price.
        assert PriceResult.model_validate({"is_estimated": True, "estimate_basis": "unavailable"}).price is None
        assert Balance.model_validate({"balance": 1.0, "currency": "USD"}).currency == "USD"


class TestResume:
    @respx.mock
    def test_timeout_carries_request_id_and_hint(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL)
        respx.get(f"{STATUS_URL}/job-9").mock(
            return_value=httpx.Response(200, json={"data": {"status": "processing", "progress": 1}})
        )
        job = client.jobs.get("job-9")
        with pytest.raises(JobTimeoutError) as ei:
            job.wait(poll_interval=0.01, max_wait=0.05)
        assert ei.value.request_id == "job-9"
        assert "still running" in str(ei.value) and "client.jobs" in str(ei.value)

    @respx.mock
    def test_jobs_get_and_status_sync(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL)
        respx.get(f"{STATUS_URL}/job-9").mock(
            return_value=httpx.Response(200, json={"data": {"status": "done", "result_url": "https://x/y.png"}})
        )
        job = client.jobs.get("job-9")
        assert job.request_id == "job-9"
        assert job.wait(poll_interval=0.01).result_url == "https://x/y.png"
        assert client.jobs.status("job-9").status == JobStatus.DONE

    @respx.mock
    async def test_jobs_async(self, async_client) -> None:  # type: ignore[no-untyped-def]
        respx.get(f"{STATUS_URL}/job-9").mock(
            return_value=httpx.Response(200, json={"data": {"status": "processing", "progress": None}})
        )
        job = async_client.jobs.get("job-9")
        assert (await async_client.jobs.status("job-9")).progress is None
        with pytest.raises(JobTimeoutError) as ei:
            await job.wait(poll_interval=0.01, max_wait=0.05)
        assert ei.value.request_id == "job-9"
