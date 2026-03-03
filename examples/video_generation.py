"""Video generation examples.

Demonstrates text-to-video, image-to-video, upscale, and background removal.

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
