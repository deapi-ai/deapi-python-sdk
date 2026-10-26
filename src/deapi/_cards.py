"""Fill unset model parameters from the model card (``info.defaults``).

The API still requires several card-driven parameters (width, height, seed, steps, ...),
so when the caller leaves one unset it is taken from the model's card. The whole
catalogue is fetched once per HTTP client (lazily, on the first call that needs it) and
cached. Defaulting never raises: if the catalogue cannot be fetched, or the model is not
in it, no card value is filled and the API answers with its normal validation error.

When the card has no default for a key but its ``limits`` pin it (``min_X == max_X``), that
value is used; nothing is derived when min and max differ.

Rules (same as the deAPI MCP server): a value the caller set always wins; a key the card
does not default is left out; free-text fields (prompt, negative_prompt, ...) are never
filled; ``seed`` falls back to ``-1`` (random) where the API requires it, even without a card.
"""

from __future__ import annotations

import threading
import time
import warnings
from typing import Any, Iterable

from deapi._exceptions import NetworkError, RateLimitError, ServerError

_PAGE_SIZE = 100
_MAX_PAGES = 50  # safety bound on pagination
_RANDOM_SEED = -1
_RETRY_COOLDOWN = 60.0  # seconds before a transient catalogue failure is retried
_CREATE_LOCK = threading.Lock()  # guards creation of a client's cache object

# Fillable keys per endpoint family (spec-required card-driven params; names match the card).
IMAGE_KEYS = ("width", "height", "steps")
VIDEO_KEYS = ("width", "height", "frames", "fps", "steps")
TTS_KEYS = ("lang", "speed", "format", "sample_rate")
MUSIC_KEYS = ("format", "inference_steps", "guidance_scale")
MUSIC_PRICE_KEYS = ("inference_steps",)

# Request field -> card limit name (``min_X``/``max_X``) where they differ.
_LIMIT_NAMES = {"inference_steps": "steps", "guidance_scale": "guidance"}

_FALSE_VALUES = ("0", "false", "no", "off")


def parse_flag(raw: str | None, default: bool = True) -> bool:
    """Parse a boolean env value: "0"/"false"/"no"/"off" disable, anything else enables."""
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in _FALSE_VALUES


def fill_from_card(
    payload: dict[str, Any], defaults: dict[str, Any], keys: Iterable[str]
) -> dict[str, Any]:
    """Return ``payload`` with each unset key in ``keys`` taken from ``defaults``.

    A key is unset when it is missing or ``None``; card values that are ``None`` are skipped.
    """
    for key in keys:
        if payload.get(key) is None and defaults.get(key) is not None:
            payload[key] = defaults[key]
    return payload


class _CardCache:
    """slug -> ``info`` dict for one HTTP client; ``failed`` after an unsuccessful fetch."""

    def __init__(self) -> None:
        self.cards: dict[str, dict[str, Any]] | None = None
        self.failed = False  # permanent failure: never retried
        self.retry_at = 0.0  # monotonic time before which a transient failure is not retried
        self.tlock = threading.Lock()  # serialises the sync fetch
        self.lock: Any = None  # asyncio.Lock, created lazily inside a running loop
        self.lock_loop: Any = None  # the loop ``lock`` belongs to

    def needs_fetch(self) -> bool:
        return self.cards is None and not self.failed and time.monotonic() >= self.retry_at

    def record_failure(self, exc: Exception) -> None:
        """Remember a failed fetch: permanent errors for good, transient ones until a cooldown ends."""
        transient = isinstance(exc, (RateLimitError, ServerError, NetworkError))
        if transient:
            self.retry_at = time.monotonic() + _RETRY_COOLDOWN
        else:
            self.failed = True
        warnings.warn(
            f"Could not load the model catalogue ({type(exc).__name__}: {exc}); model card defaults "
            "were not applied, so unset parameters (width, height, steps, ...) will be sent empty"
            + (f" (will retry in {int(_RETRY_COOLDOWN)} s)." if transient else "."),
            RuntimeWarning,
            stacklevel=4,
        )

    def values(self, model: Any, keys: Iterable[str]) -> dict[str, Any] | None:
        """Card values for ``keys``: ``info.defaults``, else a value pinned by ``info.limits``.

        A limit pins a value when ``min_X == max_X`` (e.g. fixed steps or guidance). Returns
        ``None`` when the model/catalogue is unknown.
        """
        if self.failed or self.cards is None or model not in self.cards:
            return None
        info = self.cards[model]
        defaults = info.get("defaults")
        limits = info.get("limits")
        values = dict(defaults) if isinstance(defaults, dict) else {}
        if isinstance(limits, dict):
            for key in keys:
                if values.get(key) is None:
                    name = _LIMIT_NAMES.get(key, key)
                    lo, hi = limits.get(f"min_{name}"), limits.get(f"max_{name}")
                    if lo is not None and lo == hi:
                        values[key] = lo
        return values


def _cache(client: Any) -> _CardCache:
    cache = getattr(client, "_card_cache", None)
    if cache is None:
        with _CREATE_LOCK:
            cache = getattr(client, "_card_cache", None)
            if cache is None:
                cache = _CardCache()
                client._card_cache = cache
    return cache


def _ingest(models: Any, store: dict[str, dict[str, Any]]) -> None:
    for model in models:
        slug = model.get("slug")
        if isinstance(slug, str):
            info = model.get("info")
            store[slug] = info if isinstance(info, dict) else {}


def _last_page(resp: Any) -> int:
    meta = resp.get("meta") if isinstance(resp, dict) else None
    last = meta.get("last_page") if isinstance(meta, dict) else None
    return last if isinstance(last, int) and last > 0 else 1


def _fill_tts_mode(cache: _CardCache, payload: dict[str, Any]) -> None:
    """Fill ``mode`` / ``voice`` of a TTS request from the model card.

    The API treats a missing ``mode`` as ``custom_voice`` (which then needs ``voice``), so a
    voice-design or voice-clone only model fails unless the mode is named, and a custom-voice
    model fails without a ``voice``. Explicit values always win.
    """
    if cache.failed or cache.cards is None or payload.get("model") not in cache.cards:
        return
    info = cache.cards[payload["model"]]
    features = info.get("features") if isinstance(info.get("features"), dict) else {}
    defaults = info.get("defaults") if isinstance(info.get("defaults"), dict) else {}
    if payload.get("mode") is None and features.get("supports_custom_voice") is False:
        if features.get("supports_voice_design") and not features.get("supports_voice_clone"):
            payload["mode"] = "voice_design"
        elif features.get("supports_voice_clone") and not features.get("supports_voice_design"):
            payload["mode"] = "voice_clone"
    if (payload.get("mode") or "custom_voice") == "custom_voice":
        if payload.get("voice") is None and defaults.get("voice") is not None:
            payload["voice"] = defaults["voice"]


def _apply(
    cache: _CardCache, payload: dict[str, Any], keys: Iterable[str], seed: bool, form: bool,
    tts: bool = False,
) -> dict[str, Any]:
    keys = tuple(keys)
    unset = {k for k in (*keys, "seed") if payload.get(k) is None}
    values = cache.values(payload.get("model"), keys)
    if values is not None:
        fill_from_card(payload, values, keys)
    if tts:
        _fill_tts_mode(cache, payload)
    if seed and payload.get("seed") is None:  # not card-derived: needs no catalogue
        payload["seed"] = _RANDOM_SEED
    if form:  # multipart fields are strings
        for key in unset:
            if payload.get(key) is not None:
                payload[key] = str(payload[key])
    return payload


def _wanted(client: Any, payload: dict[str, Any]) -> bool:
    return bool(getattr(client._config, "fill_model_defaults", True)) and bool(payload.get("model"))


def _needs_card(payload: dict[str, Any], keys: Iterable[str], tts: bool) -> bool:
    """True when some card-driven field is still unset, so the catalogue is worth fetching."""
    if any(payload.get(k) is None for k in keys):
        return True
    return tts and (payload.get("mode") is None or payload.get("voice") is None)


def _fetch_sync(client: Any) -> dict[str, dict[str, Any]]:
    store: dict[str, dict[str, Any]] = {}
    page = 1
    while True:
        resp = client.get(client._resolve_endpoint("models"), params={"limit": _PAGE_SIZE, "page": page})
        _ingest(resp["data"], store)
        if page >= min(_last_page(resp), _MAX_PAGES) or not resp["data"]:
            return store
        page += 1


async def _fetch_async(client: Any) -> dict[str, dict[str, Any]]:
    store: dict[str, dict[str, Any]] = {}
    page = 1
    while True:
        resp = await client.get(client._resolve_endpoint("models"), params={"limit": _PAGE_SIZE, "page": page})
        _ingest(resp["data"], store)
        if page >= min(_last_page(resp), _MAX_PAGES) or not resp["data"]:
            return store
        page += 1


def fill_payload(
    client: Any, payload: dict[str, Any], keys: Iterable[str], *, seed: bool = False, form: bool = False,
    tts: bool = False,
) -> dict[str, Any]:
    """Fill unset ``keys`` of a request payload from the model card (sync client).

    ``seed=True`` also defaults an unset ``seed`` to -1 (random). ``form=True`` stringifies
    the filled values for multipart bodies. Mutates and returns ``payload``.
    """
    if not _wanted(client, payload):
        return payload
    keys = tuple(keys)
    cache = _cache(client)
    if cache.needs_fetch() and _needs_card(payload, keys, tts):
        with cache.tlock:
            if cache.needs_fetch():
                try:
                    cache.cards = _fetch_sync(client)
                except Exception as exc:
                    cache.record_failure(exc)
    return _apply(cache, payload, keys, seed, form, tts)


async def afill_payload(
    client: Any, payload: dict[str, Any], keys: Iterable[str], *, seed: bool = False, form: bool = False,
    tts: bool = False,
) -> dict[str, Any]:
    """Async twin of :func:`fill_payload` (uses the async HTTP client)."""
    if not _wanted(client, payload):
        return payload
    keys = tuple(keys)
    cache = _cache(client)
    if cache.needs_fetch() and _needs_card(payload, keys, tts):
        import asyncio

        loop = asyncio.get_running_loop()
        if cache.lock is None or cache.lock_loop is not loop:  # an asyncio.Lock is bound to one loop
            cache.lock, cache.lock_loop = asyncio.Lock(), loop
        async with cache.lock:
            if cache.needs_fetch():
                try:
                    cache.cards = await _fetch_async(client)
                except Exception as exc:
                    cache.record_failure(exc)
    return _apply(cache, payload, keys, seed, form, tts)
