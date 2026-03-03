"""Text-to-speech and music generation examples.

Demonstrates TTS synthesis (with voice options) and music composition.

Usage:
    export DEAPI_API_KEY="sk-your-api-key"
    python examples/tts_and_audio.py
"""

from deapi import DeapiClient

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

# Uncomment to run — requires a reference audio file (3-10 seconds):
#
# print("\nCloning voice...")
# job = client.audio.synthesize(
#     text="This is generated using a cloned voice.",
#     model="Kokoro",
#     lang="en-us",
#     format="wav",
#     speed=1.0,
#     sample_rate=24000,
#     mode="voice_clone",
#     ref_audio="voice_sample.wav",   # 3-10 second reference clip
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
#     model="MusicGen",
#     duration=30,             # 30 seconds
#     inference_steps=50,
#     guidance_scale=3.0,
#     seed=42,
#     format="mp3",
#     lyrics="Dancing in the moonlight, feeling so alive",
#     bpm=128,
# )
# result = job.wait()
# print(f"Music URL: {result.result_url}")

client.close()
