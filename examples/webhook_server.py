"""Webhook server example.

Demonstrates how to receive and verify DeAPI webhook notifications
using Flask. Works with FastAPI too (see comments).

Usage:
    pip install flask
    export DEAPI_WEBHOOK_SECRET="your-webhook-secret"
    python examples/webhook_server.py

For production, use a proper WSGI server like gunicorn:
    gunicorn examples.webhook_server:app
"""

import os

from flask import Flask, Request, jsonify, request

from deapi.webhook import InvalidSignatureError, construct_event

app = Flask(__name__)
WEBHOOK_SECRET = os.environ.get("DEAPI_WEBHOOK_SECRET", "")


@app.route("/webhooks/deapi", methods=["POST"])
def handle_webhook() -> tuple:
    """Handle incoming DeAPI webhook notifications."""
    payload = request.get_data()

    # Extract security headers
    signature = request.headers.get("X-DeAPI-Signature", "")
    timestamp = request.headers.get("X-DeAPI-Timestamp", "")
    event_type = request.headers.get("X-DeAPI-Event", "")
    delivery_id = request.headers.get("X-DeAPI-Delivery-Id", "")

    print(f"Received webhook: event={event_type}, delivery={delivery_id}")

    # Verify signature and parse event
    try:
        event = construct_event(
            payload=payload,
            signature=signature,
            timestamp=timestamp,
            secret=WEBHOOK_SECRET,
            tolerance=300,  # Reject events older than 5 minutes
        )
    except InvalidSignatureError as e:
        print(f"Invalid signature: {e}")
        return jsonify({"error": "Invalid signature"}), 403

    # Handle different event types
    if event.type == "job.completed":
        print(f"Job {event.data.job_request_id} completed!")
        print(f"  Type: {event.data.job_type}")
        print(f"  Result: {event.data.result_url}")
        print(f"  Processing time: {event.data.processing_time_ms}ms")

    elif event.type == "job.processing":
        print(f"Job {event.data.job_request_id} started processing")

    elif event.type == "job.failed":
        print(f"Job {event.data.job_request_id} failed!")
        print(f"  Previous status: {event.data.previous_status}")

    return jsonify({"ok": True}), 200


if __name__ == "__main__":
    if not WEBHOOK_SECRET:
        print("WARNING: DEAPI_WEBHOOK_SECRET not set. Signature verification will fail.")
    app.run(port=8080)


# --- FastAPI version ---
#
# If you prefer FastAPI, here's the equivalent:
#
# from fastapi import FastAPI, Request, HTTPException
# from deapi.webhook import construct_event, InvalidSignatureError
#
# app = FastAPI()
#
# @app.post("/webhooks/deapi")
# async def handle_webhook(request: Request):
#     payload = await request.body()
#     try:
#         event = construct_event(
#             payload=payload,
#             signature=request.headers.get("X-DeAPI-Signature", ""),
#             timestamp=request.headers.get("X-DeAPI-Timestamp", ""),
#             secret=WEBHOOK_SECRET,
#             tolerance=300,
#         )
#     except InvalidSignatureError:
#         raise HTTPException(status_code=403, detail="Invalid signature")
#
#     if event.type == "job.completed":
#         print(f"Job {event.data.job_request_id} completed: {event.data.result_url}")
#
#     return {"ok": True}
