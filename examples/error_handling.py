"""Error handling examples.

Demonstrates how to catch and handle different error types from the SDK.

Usage:
    export DEAPI_API_KEY="sk-your-api-key"
    python examples/error_handling.py
"""

from deapi import DeapiClient
from deapi._exceptions import (
    AccountSuspendedError,
    AuthenticationError,
    DeapiError,
    InsufficientBalanceError,
    JobTimeoutError,
    NetworkError,
    RateLimitError,
    ServerError,
    ValidationError,
)

client = DeapiClient()


# --- Example 1: Catching specific errors ---

def generate_with_error_handling() -> None:
    """Show how to handle each error type."""
    try:
        job = client.images.generate(
            prompt="a beautiful landscape",
            model="Flux1schnell",
            width=1024,
            height=1024,
            seed=42,
        )
        result = job.wait(max_wait=120.0)
        print(f"Success: {result.result_url}")

    except AuthenticationError:
        print("Invalid API key. Check your DEAPI_API_KEY.")

    except AccountSuspendedError:
        print("Account suspended. Contact support.")

    except InsufficientBalanceError:
        print("Not enough credits. Top up at https://deapi.ai")

    except ValidationError as e:
        # e.errors is a dict of field -> error messages
        print(f"Invalid request: {e.message}")
        for field, messages in e.errors.items():
            print(f"  {field}: {', '.join(messages)}")

    except RateLimitError as e:
        # The SDK auto-retries 429s, but if max_retries is exceeded:
        print(f"Rate limited ({e.limit_type}). Retry after {e.retry_after}s")

    except JobTimeoutError:
        print("Job took too long. Try increasing max_wait.")

    except NetworkError:
        print("Network error. Check your connection.")

    except ServerError:
        # The SDK auto-retries 5xx, but if max_retries is exceeded:
        print("Server error. Try again later.")

    except DeapiError as e:
        # Catch-all for any other API errors
        print(f"API error {e.status_code}: {e.message}")


# --- Example 2: Check price before generating ---

def check_then_generate() -> None:
    """Check if you can afford a job before submitting it."""
    balance = client.balance()
    print(f"Current balance: ${balance.balance}")

    price = client.images.generate_price(
        prompt="a cat in space",
        model="Flux1schnell",
        width=1024,
        height=1024,
        seed=42,
    )
    print(f"Job cost: ${price.price}")

    if balance.balance < price.price:
        print("Not enough credits!")
        return

    job = client.images.generate(
        prompt="a cat in space",
        model="Flux1schnell",
        width=1024,
        height=1024,
        seed=42,
    )
    result = job.wait()
    print(f"Generated: {result.result_url}")


# --- Example 3: Manual polling with progress ---

def manual_polling() -> None:
    """Poll manually to show progress updates."""
    import time

    job = client.images.generate(
        prompt="a detailed cityscape",
        model="Flux1schnell",
        width=1024,
        height=1024,
        seed=42,
    )

    while True:
        result = job.status()
        print(f"  Status: {result.status} | Progress: {result.progress}%")

        if result.status == "done":
            print(f"  Result: {result.result_url}")
            break
        elif result.status == "error":
            print("  Job failed!")
            break

        time.sleep(2)


# --- Run examples ---

if __name__ == "__main__":
    print("=== Error Handling Example ===")
    generate_with_error_handling()

    print("\n=== Check-Then-Generate Example ===")
    check_then_generate()

    print("\n=== Manual Polling Example ===")
    manual_polling()

    client.close()
