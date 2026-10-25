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

from typing import Any, Iterable

_PAGE_SIZE = 100
_MAX_PAGES = 50  # safety bound on pagination
_RANDOM_SEED = -1

# Fillable keys per endpoint family (spec-required card-driven params; names match the card).
IMAGE_KEYS = ("width", "height", "steps")
VIDEO_KEYS = ("width", "height", "frames", "fps")
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
        self.failed = False
        self.lock: Any = None  # asyncio.Lock, created lazily inside a running loop

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


def _apply(
    cache: _CardCache, payload: dict[str, Any], keys: Iterable[str], seed: bool, form: bool
) -> dict[str, Any]:
    keys = tuple(keys)
    unset = {k for k in (*keys, "seed") if payload.get(k) is None}
    values = cache.values(payload.get("model"), keys)
    if values is not None:
        fill_from_card(payload, values, keys)
    if seed and payload.get("seed") is None:  # not card-derived: needs no catalogue
        payload["seed"] = _RANDOM_SEED
    if form:  # multipart fields are strings
        for key in unset:
            if payload.get(key) is not None:
                payload[key] = str(payload[key])
    return payload


def _wanted(client: Any, payload: dict[str, Any]) -> bool:
    return bool(getattr(client._config, "fill_model_defaults", True)) and bool(payload.get("model"))


def fill_payload(
    client: Any, payload: dict[str, Any], keys: Iterable[str], *, seed: bool = False, form: bool = False
) -> dict[str, Any]:
    """Fill unset ``keys`` of a request payload from the model card (sync client).

    ``seed=True`` also defaults an unset ``seed`` to -1 (random). ``form=True`` stringifies
    the filled values for multipart bodies. Mutates and returns ``payload``.
    """
    if not _wanted(client, payload):
        return payload
    cache = _cache(client)
    if cache.cards is None and not cache.failed:
        try:
            store: dict[str, dict[str, Any]] = {}
            page = 1
            while True:
                resp = client.get(client._resolve_endpoint("models"),
                                  params={"limit": _PAGE_SIZE, "page": page})
                _ingest(resp["data"], store)
                if page >= min(_last_page(resp), _MAX_PAGES) or not resp["data"]:
                    break
                page += 1
            cache.cards = store
        except Exception:
            cache.failed = True
    return _apply(cache, payload, keys, seed, form)


async def afill_payload(
    client: Any, payload: dict[str, Any], keys: Iterable[str], *, seed: bool = False, form: bool = False
) -> dict[str, Any]:
    """Async twin of :func:`fill_payload` (uses the async HTTP client)."""
    if not _wanted(client, payload):
        return payload
    cache = _cache(client)
    if cache.cards is None and not cache.failed:
        import asyncio

        if cache.lock is None:
            cache.lock = asyncio.Lock()
        async with cache.lock:
            if cache.cards is None and not cache.failed:
                try:
                    store: dict[str, dict[str, Any]] = {}
                    page = 1
                    while True:
                        resp = await client.get(client._resolve_endpoint("models"),
                                                params={"limit": _PAGE_SIZE, "page": page})
                        _ingest(resp["data"], store)
                        if page >= min(_last_page(resp), _MAX_PAGES) or not resp["data"]:
                            break
                        page += 1
                    cache.cards = store
                except Exception:
                    cache.failed = True
    return _apply(cache, payload, keys, seed, form)
