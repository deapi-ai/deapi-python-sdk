from __future__ import annotations

from enum import StrEnum


class JobStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    # Sent only in real-time (WebSocket) events while a job reports progress;
    # the HTTP job status endpoint keeps reporting "processing".
    IN_PROGRESS = "in_progress"
    DONE = "done"
    ERROR = "error"


class InferenceType(StrEnum):
    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
    TXT2AUDIO = "txt2audio"
    TXT2MUSIC = "txt2music"
    VIDEO2TEXT = "video2text"
    AUDIO2TEXT = "audio2text"
    IMG2TXT = "img2txt"
    IMG2VIDEO = "img2video"
    TXT2VIDEO = "txt2video"
    TXT2EMBEDDING = "txt2embedding"
    VIDEO_FILE2TEXT = "video_file2text"
    AUDIO_FILE2TEXT = "audio_file2text"
    IMAGE_UPSCALE = "img-upscale"
    IMAGE_REMOVE_BG = "img-rmbg"
    VIDEO_UPSCALE = "vid-upscale"
    VIDEO_REMOVE_BG = "vid-rmbg"
    AUDIO2VIDEO = "audio2video"
    VIDEO2DESC = "video2desc"
    VIDEO_FILE2DESC = "video_file2desc"
