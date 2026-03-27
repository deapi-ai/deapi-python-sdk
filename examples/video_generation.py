"""Video generation examples.

Demonstrates text-to-video, image-to-video, audio-to-video, video replace,
upscale, and background removal.

Usage:
    export DEAPI_API_KEY="sk-your-api-key"
    python examples/video_generation.py
"""

from deapi import DeapiClient

client = DeapiClient()

# --- Text-to-video ---

print("Generating video from text...")
job = client.video.generate(
    prompt="a rocket launching into space, cinematic, slow motion",
    negative_prompt="blurry, low quality",
    model="Ltx2_19B_Dist_FP8",
    width=768,
    height=512,
    steps=1,
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
#     first_frame_image="landscape.jpg",
#     model="Ltx2_19B_Dist_FP8",
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
#     first_frame_image="daytime.jpg",
#     last_frame_image="nighttime.jpg",
#     model="Ltx2_19B_Dist_FP8",
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
#     audio="track.mp3",
#     model="ltx-audio2video",
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
#     audio="track.mp3",
#     first_frame_image="start_frame.png",
#     last_frame_image="end_frame.png",
#     model="ltx-audio2video",
#     width=512,
#     height=512,
#     seed=42,
#     frames=97,
#     fps=24,
#     guidance=7.5,
#     steps=20,
# )

# --- Video replace (swap character) ---

# Uncomment to run — requires a video and a reference image:
#
# print("\nReplacing character in video...")
# job = client.video.replace(
#     video="input.mp4",
#     ref_image="reference_face.png",
#     model="vid-replace-model",
#     prompt="replace the character",
# )
# result = job.wait()
# print(f"Video replace: {result.result_url}")

# Price calculation for video replace:
#
# price = client.video.replace_price(
#     model="vid-replace-model",
#     duration=10.5,  # or pass video=b"..." to auto-detect duration
#     width=512,
#     height=512,
# )
# print(f"Replace price: ${price.price}")

# --- Video upscale ---

# Uncomment to run — requires a video file:
#
# print("\nUpscaling video...")
# job = client.video.upscale(video="low_res.mp4", model="vid-upscale-model")
# result = job.wait()
# print(f"Upscaled: {result.result_url}")

# --- Video background removal ---

# Uncomment to run — requires a video file:
#
# print("\nRemoving video background...")
# job = client.video.remove_background(video="greenscreen.mp4", model="vid-rmbg-model")
# result = job.wait()
# print(f"Background removed: {result.result_url}")

# --- Price calculation ---

price = client.video.generate_price(
    prompt="a rocket launching",
    model="Ltx2_19B_Dist_FP8",
    width=768,
    height=512,
    steps=1,
    seed=42,
    frames=120,
    fps=24,
)
print(f"\nVideo generation price: ${price.price}")

client.close()
