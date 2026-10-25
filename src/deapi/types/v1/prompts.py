from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Inference types the v2 prompt booster accepts (v2 dot notation). Upscaling, background
# removal, OCR, transcription and embeddings have no booster and are rejected with 422.
PromptBoosterType = Literal[
    "images.generations",
    "images.edits",
    "videos.generations",
    "videos.animations",
    "audio.speech",
    "audio.music",
]

# Types that need a reference ``image`` alongside the prompt.
PROMPT_BOOSTER_TYPES_REQUIRING_IMAGE = ("images.edits", "videos.animations")


class EnhancePromptResult(BaseModel):
    """Result from prompt enhancement endpoints (image, video, image2image)."""
    prompt: str
    negative_prompt: str | None = None


class EnhanceSpeechPromptResult(BaseModel):
    """Result from speech prompt enhancement (no negative_prompt)."""
    prompt: str


class SamplePromptResult(BaseModel):
    """Result from sample prompts endpoint."""
    type: str
    prompt: str
