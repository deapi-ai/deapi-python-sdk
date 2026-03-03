"""Transcription examples.

Demonstrates video URL, audio file, and unified transcription endpoints.

Usage:
    export DEAPI_API_KEY="sk-your-api-key"
    python examples/transcription.py
"""

from deapi import DeapiClient

client = DeapiClient()

# --- Unified endpoint (recommended) — URL ---

print("Transcribing video from URL...")
job = client.transcription.create(
    source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    include_ts=True,
    model="WhisperLargeV3",
    return_result_in_response=True,
)
result = job.wait()
print(f"Status: {result.status}")
if result.result:
    # When return_result_in_response=True, text is inline
    print(f"Transcription (first 500 chars):\n{result.result[:500]}")
else:
    print(f"Result URL: {result.result_url}")

# --- Unified endpoint — file upload ---

# Uncomment to run — requires an audio/video file:
#
# print("\nTranscribing audio file...")
# job = client.transcription.create(
#     source_file="interview.mp3",   # accepts str path, Path, bytes, or file-like
#     include_ts=False,
#     model="WhisperLargeV3",
#     return_result_in_response=True,
# )
# result = job.wait()
# print(f"Transcription: {result.result}")

# --- Price calculation ---

print("\nCalculating transcription price...")
price = client.transcription.create_price(
    source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    include_ts=True,
    model="WhisperLargeV3",
)
print(f"Cost: ${price.price}")

# Or estimate by duration (skip URL metadata fetch):
price = client.transcription.create_price(
    include_ts=True,
    model="WhisperLargeV3",
    duration_seconds=120.0,
)
print(f"Cost for 2 min audio: ${price.price}")

# --- Legacy endpoints (if you need specific control) ---

# Video URL (YouTube, Twitch, Twitter, TikTok, Kick):
# job = client.transcription.from_video_url(
#     video_url="https://youtube.com/watch?v=...",
#     include_ts=True,
#     model="WhisperLargeV3",
# )

# Audio URL (Twitter Spaces only):
# job = client.transcription.from_audio_url(
#     audio_url="https://twitter.com/i/spaces/...",
#     include_ts=True,
#     model="WhisperLargeV3",
# )

# Audio file upload:
# job = client.transcription.from_audio_file(
#     audio="recording.wav",
#     include_ts=False,
#     model="WhisperLargeV3",
# )

# Video file upload:
# job = client.transcription.from_video_file(
#     video="meeting.mp4",
#     include_ts=True,
#     model="WhisperLargeV3",
# )

client.close()
