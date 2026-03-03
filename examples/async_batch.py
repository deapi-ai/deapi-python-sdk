"""Async batch processing example.

Demonstrates submitting multiple jobs concurrently with AsyncDeapiClient.

Usage:
    export DEAPI_API_KEY="sk-your-api-key"
    python examples/async_batch.py
"""

import asyncio

from deapi import AsyncDeapiClient


async def main() -> None:
    async with AsyncDeapiClient() as client:
        # Check balance before submitting
        balance = await client.balance()
        print(f"Account balance: ${balance.balance}")

        # Submit multiple image generation jobs concurrently
        prompts = [
            "a mountain landscape at golden hour",
            "an ocean sunset with dramatic clouds",
            "a misty forest path in autumn",
            "a futuristic city skyline at night",
        ]

        print(f"\nSubmitting {len(prompts)} jobs concurrently...")
        jobs = await asyncio.gather(*[
            client.images.generate(
                prompt=prompt,
                model="Flux1schnell",
                width=1024,
                height=1024,
                seed=i + 1,
            )
            for i, prompt in enumerate(prompts)
        ])

        for job in jobs:
            print(f"  Submitted: {job.request_id}")

        # Wait for all results concurrently
        print("\nWaiting for results...")
        results = await asyncio.gather(*[job.wait() for job in jobs])

        for i, result in enumerate(results):
            print(f"\n  Job {i + 1}: {prompts[i]}")
            print(f"    Status: {result.status}")
            print(f"    URL: {result.result_url}")


asyncio.run(main())
