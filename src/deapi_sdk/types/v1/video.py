from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict


class Text2VideoParams(BaseModel):
    prompt: str
    model: str
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    seed: int | None = None
    frames: int | None = None
    fps: int | None = None
    negative_prompt: str | None = None
    guidance: float | None = None
    enhance_prompt: bool | None = None
    webhook_url: str | None = None
    webhook_secret: str | None = None


class Img2VideoParams(BaseModel):
    prompt: str
    model: str
    seed: int | None = None
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    fps: int | None = None
    negative_prompt: str | None = None
    guidance: float | None = None
    steps: int | None = None
    enhance_prompt: bool | None = None
    webhook_url: str | None = None
    webhook_secret: str | None = None


class Aud2VideoParams(BaseModel):
    prompt: str
    model: str
    width: int | None = None
    height: int | None = None
    seed: int | None = None
    frames: int | None = None
    fps: int | None = None
    negative_prompt: str | None = None
    guidance: float | None = None
    steps: int | None = None
    webhook_url: str | None = None
    webhook_secret: str | None = None


class VideoReplaceParams(BaseModel):
    model: str
    prompt: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    seed: int | None = None
    webhook_url: str | None = None
    webhook_secret: str | None = None


class VideoEvent(BaseModel):
    """One timestamped on-screen event (``start`` / ``end`` in seconds)."""

    model_config = ConfigDict(extra="allow")

    start: float
    end: float
    description: str
    merged: bool | None = None
    edge_distance: float | None = None


class VideoDescription(BaseModel):
    """Result of a video description job: ``events``, ``scenes`` and source ``meta``."""

    model_config = ConfigDict(extra="allow")

    events: list[VideoEvent] = []
    scenes: list[Any] = []
    meta: dict[str, Any] = {}


def parse_video_description(raw: str | bytes | dict[str, Any]) -> VideoDescription:
    """Parse a video description result (JSON string from ``JobResult.result``, bytes, or dict)."""
    if isinstance(raw, (str, bytes)):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise ValueError("Video description must be a JSON object")
    return VideoDescription.model_validate(raw)
