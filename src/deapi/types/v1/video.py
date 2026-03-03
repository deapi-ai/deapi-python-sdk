from __future__ import annotations

from pydantic import BaseModel


class Text2VideoParams(BaseModel):
    prompt: str
    model: str
    width: int
    height: int
    steps: int
    seed: int
    frames: int
    fps: int
    negative_prompt: str | None = None
    guidance: float | None = None
    webhook_url: str | None = None


class Img2VideoParams(BaseModel):
    prompt: str
    model: str
    seed: int
    width: int
    height: int
    frames: int
    fps: int
    negative_prompt: str | None = None
    guidance: float | None = None
    steps: int | None = None
    webhook_url: str | None = None
