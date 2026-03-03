"""Basic image generation examples.

Demonstrates text-to-image, image-to-image, upscale, and background removal.

Usage:
    export DEAPI_API_KEY="sk-your-api-key"
    python examples/basic_image.py
"""

from deapi import DeapiClient

client = DeapiClient()

# --- Check price before generating ---

price = client.images.generate_price(
    prompt="a cat floating in a nebula, photorealistic",
    model="Flux1schnell",
    width=1024,
    height=1024,
    seed=42,
)
print(f"Generation will cost: ${price.price}")

# --- Text-to-image ---

print("\nGenerating image...")
job = client.images.generate(
    prompt="a cat floating in a nebula, photorealistic",
    negative_prompt="blurry, low quality, distorted",
    model="Flux1schnell",
    width=1024,
    height=1024,
    seed=42,
)
print(f"Job submitted: {job.request_id}")

result = job.wait()
print(f"Status: {result.status}")
print(f"Image URL: {result.result_url}")
if result.results_alt_formats:
    print(f"Alt formats: {result.results_alt_formats}")

# --- Image-to-image (transform an existing image) ---

# Uncomment to run — requires an image file:
#
# print("\nTransforming image...")
# job = client.images.transform(
#     prompt="make it look like a watercolor painting",
#     image="photo.jpg",          # accepts str path, Path, bytes, or file-like
#     model="QwenImageEdit_Plus_NF4",
#     steps=30,
#     seed=42,
# )
# result = job.wait()
# print(f"Transformed: {result.result_url}")

# --- Image upscale ---

# Uncomment to run — requires an image file:
#
# print("\nUpscaling image...")
# job = client.images.upscale(image="low_res.png", model="RealESRGAN_x4")
# result = job.wait()
# print(f"Upscaled: {result.result_url}")

# --- Background removal ---

# Uncomment to run — requires an image file:
#
# print("\nRemoving background...")
# job = client.images.remove_background(image="photo.png", model="Ben2")
# result = job.wait()
# print(f"Background removed: {result.result_url}")

client.close()
