"""Text-to-speech and music generation examples.

Demonstrates TTS synthesis (with voice options) and music composition.

Usage:
    export DEAPI_API_KEY="sk-dp-<id>-<secret>"
    python examples/tts_and_audio.py
"""

from pathlib import Path  # noqa: F401 - used by the commented-out examples below

from deapi_sdk import DeapiClient

client = DeapiClient()

# --- Text-to-speech ---

print("Generating speech...")
job = client.audio.synthesize(
    text="Hello! Welcome to DeAPI, the distributed AI inference platform.",
    model="Kokoro",
    voice="af_sky",
    lang="en-us",
    format="mp3",
    speed=1.0,
    sample_rate=24000,
)
result = job.wait()
print(f"Audio URL: {result.result_url}")

# --- TTS with voice cloning ---

# Uncomment to run — requires a reference audio file (5-15 seconds). Voice cloning needs a
# model whose card has supports_voice_clone (here Qwen3_TTS_12Hz_1_7B_Base; Kokoro does not clone).
#
# print("\nCloning voice...")
# job = client.audio.synthesize(
#     text="This is generated using a cloned voice.",
#     model="Qwen3_TTS_12Hz_1_7B_Base",
#     lang="English",
#     format="mp3",
#     mode="voice_clone",
#     ref_audio=Path("voice_sample.wav"),   # 5-15 second reference clip
#     ref_text="The original text spoken in the reference.",
# )
# result = job.wait()
# print(f"Cloned voice audio: {result.result_url}")

# --- TTS price calculation ---

price = client.audio.synthesize_price(
    model="Kokoro",
    lang="en-us",
    format="mp3",
    speed=1.0,
    sample_rate=24000,
    text="Hello! Welcome to DeAPI.",
)
print(f"\nTTS price: ${price.price}")

# You can also estimate price by character count:
price = client.audio.synthesize_price(
    model="Kokoro",
    lang="en-us",
    format="mp3",
    speed=1.0,
    sample_rate=24000,
    count_text=500,  # 500 characters
)
print(f"Price for 500 chars: ${price.price}")

# --- Music generation ---

# Uncomment to run:
#
# print("\nComposing music...")
# job = client.audio.compose(
#     caption="upbeat electronic track with synth leads and a driving bassline",
#     model="AceStep_1_5_XL_Turbo_INT8",
#     duration=30,             # 30 seconds (10-300)
#     inference_steps=8,       # fixed for this model
#     guidance_scale=1.0,      # fixed for this model
#     seed=42,
#     format="mp3",
#     lyrics="Dancing in the moonlight, feeling so alive",
#     bpm=128,
# )
# result = job.wait()
# print(f"Music URL: {result.result_url}")

client.close()
