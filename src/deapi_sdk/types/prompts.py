from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Inference types the prompt booster accepts (dot notation). Upscaling, background
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
    """Result of the prompt enhancement endpoint."""
    prompt: str
    negative_prompt: str | None = None

