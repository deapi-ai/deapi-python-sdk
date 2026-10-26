"""Download a finished job's result from its storage URL.

A plain GET: no Authorization header (the API key must never reach a storage host), redirects
followed, the client's timeout. HTTP errors become one clear ``DeapiError`` (never the
API-key-specific auth errors, which would mislead), transport errors become ``NetworkError``.
"""

from __future__ import annotations

from typing import Any

import httpx

from deapi._exceptions import DeapiError, NetworkError


def _check(resp: httpx.Response) -> str:
    if resp.status_code >= 400:
        raise DeapiError(f"Could not download result (HTTP {resp.status_code})", status_code=resp.status_code)
    return resp.text


def download_text(url: str, timeout: Any) -> str:
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    except httpx.TransportError as exc:
        raise NetworkError(str(exc)) from exc
    return _check(resp)


async def adownload_text(url: str, timeout: Any) -> str:
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as http:
            resp = await http.get(url)
    except httpx.TransportError as exc:
        raise NetworkError(str(exc)) from exc
    return _check(resp)
