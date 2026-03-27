from __future__ import annotations


# Endpoint paths per API version.
# Resource classes resolve paths via ENDPOINTS[version][operation_name].
# When v2 arrives, add a "v2" key with its own mapping.

ENDPOINTS: dict[str, dict[str, str]] = {
    "v1": {
        # Images
        "txt2img": "txt2img",
        "txt2img_price": "txt2img/price-calculation",
        "img2img": "img2img",
        "img2img_price": "img2img/price-calculation",
        "img_upscale": "img-upscale",
        "img_upscale_price": "img-upscale/price-calculation",
        "img_rmbg": "img-rmbg",
        "img_rmbg_price": "img-rmbg/price-calculation",
        # Audio
        "txt2audio": "txt2audio",
        "txt2audio_price": "txt2audio/price-calculation",
        "txt2music": "txt2music",
        "txt2music_price": "txt2music/price-calculation",
        # Video
        "txt2video": "txt2video",
        "txt2video_price": "txt2video/price-calculation",
        "img2video": "img2video",
        "img2video_price": "img2video/price-calculation",
        "vid_upscale": "vid-upscale",
        "vid_upscale_price": "vid-upscale/price-calculation",
        "vid_rmbg": "vid-rmbg",
        "vid_rmbg_price": "vid-rmbg/price-calculation",
        "aud2video": "aud2video",
        "aud2video_price": "aud2video/price-calculation",
        "vid_replace": "videos/replace",
        "vid_replace_price": "videos/replace/price",
        # Transcription
        "transcribe": "transcribe",
        "transcribe_price": "transcribe/price-calculation",
        "vid2txt": "vid2txt",
        "vid2txt_price": "vid2txt/price-calculation",
        "aud2txt": "aud2txt",
        "aud2txt_price": "aud2txt/price-calculation",
        "audiofile2txt": "audiofile2txt",
        "audiofile2txt_price": "audiofile2txt/price-calculation",
        "videofile2txt": "videofile2txt",
        "videofile2txt_price": "videofile2txt/price-calculation",
        # Embeddings
        "txt2embedding": "txt2embedding",
        "txt2embedding_price": "txt2embedding/price-calculation",
        # Image-to-text (OCR)
        "img2txt": "img2txt",
        "img2txt_price": "img2txt/price-calculation",
        # Prompts
        "prompt_image": "prompt/image",
        "prompt_image_price": "prompt/image/price-calculation",
        "prompt_video": "prompt/video",
        "prompt_video_price": "prompt/video/price-calculation",
        "prompt_speech": "prompt/speech",
        "prompt_speech_price": "prompt/speech/price-calculation",
        "prompt_image2image": "prompt/image2image",
        "prompt_image2image_price": "prompt/image2image/price-calculation",
        "prompt_samples": "prompts/samples",
        "prompt_samples_price": "prompts/samples/price-calculation",
        # Account
        "balance": "balance",
        "models": "models",
        "request_status": "request-status",
    },
}
