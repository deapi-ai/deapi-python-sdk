from __future__ import annotations

from typing import AsyncGenerator, Generator

import pytest

from deapi import AsyncDeapiClient, DeapiClient

TEST_BASE_URL = "https://test.deapi.ai"
TEST_API_KEY = "test-key-123"


@pytest.fixture
def client() -> Generator[DeapiClient, None, None]:
    with DeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL) as c:
        yield c


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncDeapiClient, None]:
    async with AsyncDeapiClient(api_key=TEST_API_KEY, base_url=TEST_BASE_URL) as c:
        yield c
