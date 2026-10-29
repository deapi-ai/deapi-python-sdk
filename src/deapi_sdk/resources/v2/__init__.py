from __future__ import annotations

# Most v2 resources are byte-identical to v1 except for endpoint paths,
# which are version-aware via ENDPOINTS["v2"]. We re-export the v1 classes
# for those, and define dedicated v2 classes only where behavior diverges
# (Transcription consolidated, Prompts unified-or-deprecated).
from deapi_sdk.resources.v1.audio import AsyncAudio, Audio
from deapi_sdk.resources.v1.embeddings import AsyncEmbeddings, Embeddings
from deapi_sdk.resources.v1.images import AsyncImages, Images
from deapi_sdk.resources.v1.models import AsyncModels, Models
from deapi_sdk.resources.v1.ocr import OCR, AsyncOCR
from deapi_sdk.resources.v1.video import AsyncVideo, Video
from deapi_sdk.resources.v2.account import AccountResource, AsyncAccountResource
from deapi_sdk.resources.v2.keys import AsyncKeysResource, KeysResource
from deapi_sdk.resources.v2.prompts import AsyncPrompts, Prompts
from deapi_sdk.resources.v2.transcription import AsyncTranscription, Transcription

__all__ = [
    "AccountResource",
    "AsyncAccountResource",
    "AsyncKeysResource",
    "KeysResource",
    "AsyncAudio",
    "AsyncEmbeddings",
    "AsyncImages",
    "AsyncModels",
    "AsyncOCR",
    "AsyncPrompts",
    "AsyncTranscription",
    "AsyncVideo",
    "Audio",
    "Embeddings",
    "Images",
    "Models",
    "OCR",
    "Prompts",
    "Transcription",
    "Video",
]
