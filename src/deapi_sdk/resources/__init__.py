from __future__ import annotations

from deapi_sdk.resources.account import AccountResource, AsyncAccountResource
from deapi_sdk.resources.audio import AsyncAudio, Audio
from deapi_sdk.resources.embeddings import AsyncEmbeddings, Embeddings
from deapi_sdk.resources.images import AsyncImages, Images
from deapi_sdk.resources.keys import AsyncKeysResource, KeysResource
from deapi_sdk.resources.models import AsyncModels, Models
from deapi_sdk.resources.ocr import OCR, AsyncOCR
from deapi_sdk.resources.prompts import AsyncPrompts, Prompts
from deapi_sdk.resources.transcription import AsyncTranscription, Transcription
from deapi_sdk.resources.video import AsyncVideo, Video

__all__ = [
    "OCR",
    "AccountResource",
    "AsyncAccountResource",
    "AsyncAudio",
    "AsyncEmbeddings",
    "AsyncImages",
    "AsyncKeysResource",
    "AsyncModels",
    "AsyncOCR",
    "AsyncPrompts",
    "AsyncTranscription",
    "AsyncVideo",
    "Audio",
    "Embeddings",
    "Images",
    "KeysResource",
    "Models",
    "Prompts",
    "Transcription",
    "Video",
]
