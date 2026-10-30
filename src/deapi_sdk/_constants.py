from __future__ import annotations

# The SDK targets deAPI API v2. Every endpoint lives under this prefix.
API_VERSION = "v2"
API_PREFIX = "/api/v2"

# Endpoint paths relative to API_PREFIX, keyed by a stable operation name.
ENDPOINTS: dict[str, str] = {
    # Images
    "txt2img": "images/generations",
    "txt2img_price": "images/generations/price",
    "img2img": "images/edits",
    "img2img_price": "images/edits/price",
    "img_upscale": "images/upscales",
    "img_upscale_price": "images/upscales/price",
    "img_rmbg": "images/background-removals",
    "img_rmbg_price": "images/background-removals/price",
    # Audio
    "txt2audio": "audio/speech",
    "txt2audio_price": "audio/speech/price",
    "txt2music": "audio/music",
    "txt2music_price": "audio/music/price",
    # Video
    "txt2video": "videos/generations",
    "txt2video_price": "videos/generations/price",
    "img2video": "videos/animations",
    "img2video_price": "videos/animations/price",
    "vid_upscale": "videos/upscales",
    "vid_upscale_price": "videos/upscales/price",
    "vid_rmbg": "videos/background-removals",
    "vid_rmbg_price": "videos/background-removals/price",
    "aud2video": "videos/audio-syncs",
    "aud2video_price": "videos/audio-syncs/price",
    "vid_replace": "videos/replacements",
    "vid_replace_price": "videos/replacements/price",
    "vid_description": "videos/descriptions",
    "vid_description_price": "videos/descriptions/price",
    # Transcription (unified — single endpoint replaces all v1 transcription routes)
    "transcribe": "audio/transcriptions",
    "transcribe_price": "audio/transcriptions/price",
    # Embeddings
    "txt2embedding": "embeddings",
    "txt2embedding_price": "embeddings/price",
    # Image-to-text (OCR)
    "img2txt": "images/ocr",
    "img2txt_price": "images/ocr/price",
    # Prompts (unified enhancer — replaces all v1 prompt booster routes)
    "prompts_enhance": "prompts/enhancements",
    "prompts_enhance_price": "prompts/enhancements/price",
    # Account (ADMIN key only — an API-role key gets 403 insufficient_permissions)
    "balance": "account/balance",
    "account": "account",
    "account_usage": "account/usage",
    "account_payments": "account/payments",
    # Keys — the calling key (any role)
    "key": "key",
    "key_usage": "key/usage",
    # Keys — account key management (ADMIN key only); per-key paths are keys/{id}
    "keys": "keys",
    # Catalogue and jobs
    "models": "models",
    "request_status": "jobs",
}
