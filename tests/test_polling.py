from __future__ import annotations

import httpx
import pytest
import respx

from deapi import DeapiClient
from deapi._exceptions import JobTimeoutError
from deapi._types import JobStatus

TEST_BASE_URL = "https://test.deapi.ai"
TEST_API_KEY = "test-key-123"
STATUS_URL = f"{TEST_BASE_URL}/api/v1/client/request-status"

SUBMIT_RESPONSE = {"data": {"request_id": "job-123"}}


class TestPolling:
    @respx.mock
    def test_job_status_single_poll(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL)
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2img").mock(
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
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL)
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2img").mock(
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
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL)
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2img").mock(
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
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL)
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2img").mock(
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
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL)
        respx.post(f"{TEST_BASE_URL}/api/v1/client/txt2img").mock(
            return_value=httpx.Response(200, json=SUBMIT_RESPONSE)
        )
        job = client.images.generate(prompt="test", model="sdxl", width=512, height=512, seed=1)
        assert repr(job) == "Job(request_id='job-123')"


class TestModels:
    @respx.mock
    def test_list_models(self) -> None:
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL)
        respx.get(f"{TEST_BASE_URL}/api/v1/client/models").mock(
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
        client = DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL)
        route = respx.get(f"{TEST_BASE_URL}/api/v1/client/models").mock(
            return_value=httpx.Response(200, json={
                "data": [{"name": "sdxl", "slug": "sdxl", "inference_types": ["txt2img"], "info": {}}],
                "links": {"first": "...", "last": "...", "prev": None, "next": None},
                "meta": {"current_page": 1, "last_page": 1, "per_page": 15, "total": 1},
            })
        )
        client.models.list(inference_types=["txt2img", "img2img"])
        request = route.calls.last.request
        assert "filter%5Binference_types%5D=txt2img%2Cimg2img" in str(request.url)
