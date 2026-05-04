from __future__ import annotations

from typing import AsyncGenerator, Generator

import pytest

from deapi import AsyncDeapiClient, DeapiClient

TEST_BASE_URL = "https://test.deapi.ai"
TEST_API_KEY = "test-key-123"


# v1 client — explicit since the SDK now defaults to v2.
# Existing v1 test files use these fixture names.
@pytest.fixture
def client() -> Generator[DeapiClient, None, None]:
    with DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, api_version="v1") as c:
        yield c


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncDeapiClient, None]:
    async with AsyncDeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, api_version="v1") as c:
        yield c


# v2 client — for v2 test modules.
@pytest.fixture
def client_v2() -> Generator[DeapiClient, None, None]:
    with DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, api_version="v2") as c:
        yield c


@pytest.fixture
async def async_client_v2() -> AsyncGenerator[AsyncDeapiClient, None]:
    async with AsyncDeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL, api_version="v2") as c:
        yield c
