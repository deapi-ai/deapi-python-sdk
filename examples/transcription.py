"""Transcription examples.

Demonstrates the unified transcription endpoint and both transcript formats.

Usage:
    export DEAPI_API_KEY="sk-dp-<id>-<secret>"
    python examples/transcription.py
"""

from deapi_sdk import DeapiClient

client = DeapiClient()

# --- Two transcript formats ---
#
# Models whose card has info.limits.output_format == "json" (e.g. WhisperLargeV3Ct2)
# always return a JSON document (text, language, segments with start/end/speaker/words).
# Models with output_format == null (e.g. WhisperLargeV3) return plain text with
# timestamps written inline. get_transcript() handles both and returns a Transcript.


def show(model: str, **options) -> None:
    job = client.transcription.create(
        source_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        model=model,
        include_ts=True,
        **options,
    )
    result = job.wait()
    # Uses the inline result if present, otherwise downloads result_url
    # (without sending your API key to the storage host).
    transcript = client.transcription.get_transcript(result)
    print(f"{model}: format={transcript.format}")
    print(transcript.text[:300])
    for seg in transcript.segments[:3]:  # empty for format == "text"
        print(f"  [{seg.start} - {seg.end}] {seg.speaker or ''} {seg.text}")


print("JSON-format model (segments, optional diarization):")
show("WhisperLargeV3Ct2", ts_level="segment", diarize=True, lang="en")

print("\nPlain-text model (timestamps inline in the text):")
show("WhisperLargeV3")

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
