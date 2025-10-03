#!/usr/bin/env python3
"""
Example: Real-time price streaming with 254Carbon Python SDK

Tips:
- Ensure WebSocket endpoint is reachable (local_dev=True or valid base_url)
- Use callback for legacy sync patterns or consume the async generator
"""

import asyncio
from datetime import datetime, timedelta
from carbon254 import CarbonClient

def price_callback(price_tick):
    """Handle incoming price updates."""
    print(f"📈 {price_tick.instrument_id}: ${price_tick.value:.2f} at {price_tick.event_time}")

async def main():
    """Demo real-time price streaming."""
    print("🚀 Starting real-time price streaming demo...")

    # Initialize client for local development
    client = CarbonClient(local_dev=True)

    # Instruments to monitor
    instruments = ["MISO.HUB.INDIANA", "PJM.HUB.WEST", "CAISO.SP15"]

    print(f"📡 Subscribing to: {', '.join(instruments)}")
    print("💡 Streaming for 30 seconds... (Ctrl+C to stop)")

    try:
        # Stream prices for 30 seconds
        await asyncio.wait_for(
            client.stream_prices(instruments, price_callback),
            timeout=30
        )
    except asyncio.TimeoutError:
        print("⏰ Demo completed after 30 seconds")
    except KeyboardInterrupt:
        print("👋 Streaming stopped by user")

    client.close()

if __name__ == "__main__":
    asyncio.run(main())
