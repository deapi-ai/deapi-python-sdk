"""Real-time job status over WebSocket.

Opens one WebSocket connection, submits an image generation and follows its progress live.
With --watch it just prints every status event on your channel (all of your jobs).

Requires the optional extra:
    pip install "deapi-sdk[ws] @ git+https://github.com/deapi-ai/deapi-python-sdk.git"

Usage:
    export DEAPI_API_KEY="sk-dp-<id>-<secret>"
    export DEAPI_CLIENT_ID="your-client-id"      # the "Client ID" shown in your deAPI dashboard
    python examples/realtime_status.py
    python examples/realtime_status.py --watch
"""

import sys

from deapi_sdk import DeapiClient, JobStatusEvent


def print_progress(event: JobStatusEvent) -> None:
    progress = f"{event.progress:.0f}%" if event.progress is not None else "..."
    print(f"  {event.status}: {progress}")


def main() -> None:
    # Open the connection BEFORE submitting jobs so no event is missed.
    # client_id falls back to DEAPI_CLIENT_ID.
    with DeapiClient() as client, client.realtime() as rt:
        if "--watch" in sys.argv:
            print("Watching job events (Ctrl+C to stop)...")
            try:
                for event in rt.events():
                    print(f"{event.request_id}: {event.status} {event.progress}")
            except KeyboardInterrupt:
                pass
            return

        print("Generating image...")
        job = client.images.generate(
            prompt="a cat floating in a nebula, photorealistic",
            negative_prompt="blurry, low quality, distorted",
            model="Flux1schnell",
            width=1024,
            height=1024,
            seed=42,
        )
        print(f"Job submitted: {job.request_id}")

        result = rt.wait(job, on_progress=print_progress)
        print(f"Status: {result.status}")
        print(f"Image URL: {result.result_url}")


main()
