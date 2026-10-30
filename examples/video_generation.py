"""Video generation examples.

Demonstrates text-to-video, image-to-video, audio-to-video and upscale.

Usage:
    export DEAPI_API_KEY="sk-dp-<id>-<secret>"
    python examples/video_generation.py
"""

from pathlib import Path  # noqa: F401 - used by the commented-out examples below

from deapi_sdk import DeapiClient

client = DeapiClient()

# --- Text-to-video ---

print("Generating video from text...")
job = client.video.generate(
    prompt="a rocket launching into space, cinematic, slow motion",
    model="Ltx2_5_22B_Dist_INT8",
    width=768,
    height=512,
    seed=42,
    frames=120,
    fps=24,
)
print(f"Job submitted: {job.request_id}")

result = job.wait()
print(f"Status: {result.status}")
print(f"Video URL: {result.result_url}")

# --- Image-to-video (animate a still image) ---

# Uncomment to run — requires an image file:
#
# print("\nAnimating image...")
# job = client.video.animate(
#     prompt="gentle camera pan across the landscape with clouds moving",
#     first_frame_image=Path("landscape.jpg"),
#     model="Ltx2_5_22B_Dist_INT8",
#     seed=42,
#     width=768,
#     height=512,
#     frames=120,
#     fps=24,
# )
# result = job.wait()
# print(f"Animated video: {result.result_url}")

# --- With first and last frame (for controlled transitions) ---

# Uncomment to run — requires two image files:
#
# job = client.video.animate(
#     prompt="smooth transition from day to night",
#     first_frame_image=Path("daytime.jpg"),
#     last_frame_image=Path("nighttime.jpg"),
#     model="Ltx2_5_22B_Dist_INT8",
#     seed=42,
#     width=768,
#     height=512,
#     frames=120,
#     fps=24,
# )

# --- Audio-to-video ---

# Uncomment to run — requires an audio file:
#
# print("\nGenerating video from audio...")
# job = client.video.generate_from_audio(
#     prompt="abstract shapes pulsing to the beat",
#     audio=Path("track.mp3"),
#     model="Ltx2_5_22B_Dist_INT8",
#     width=512,
#     height=512,
#     seed=42,
#     frames=97,
#     fps=24,
# )
# result = job.wait()
# print(f"Audio-to-video: {result.result_url}")

# With optional first/last frame images for more control:
#
# job = client.video.generate_from_audio(
#     prompt="music visualization",
#     audio=Path("track.mp3"),
#     first_frame_image=Path("start_frame.png"),
#     last_frame_image=Path("end_frame.png"),
#     model="Ltx2_5_22B_Dist_INT8",
#     width=512,
#     height=512,
#     seed=42,
#     frames=97,
#     fps=24,
# )

# --- Video upscale ---

# Uncomment to run — requires a video file:
#
# print("\nUpscaling video...")
# job = client.video.upscale(video=Path("low_res.mp4"), model="FlashVSR_Tiny", scale=2)
# result = job.wait()
# print(f"Upscaled: {result.result_url}")

# --- Price calculation ---

price = client.video.generate_price(
    prompt="a rocket launching",
    model="Ltx2_5_22B_Dist_INT8",
    width=768,
    height=512,
    seed=42,
    frames=120,
    fps=24,
)
print(f"\nVideo generation price: ${price.price}")

client.close()
