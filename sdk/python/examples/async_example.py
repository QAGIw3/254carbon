#!/usr/bin/env python3
"""
Example: Enhanced async functionality with 254Carbon Python SDK

Demonstrates:
- Async/await API calls with retry/backoff
- Real-time streaming with async generators
- Connection health monitoring and reconnection

Prerequisites:
- Install: `pip install carbon254` (or run from repo root)
- Use local_dev=True for mock fallbacks, or set a valid API key
"""
import asyncio
import pandas as pd
from datetime import datetime, timedelta, date
from carbon254 import CarbonClient


async def demo_async_api_calls():
    """Demo async API calls with retry logic."""
    print("🚀 Async API Calls Demo")

    # Initialize client
    client = CarbonClient(local_dev=True)

    try:
        # Async instruments call
        print("📋 Getting instruments asynchronously...")
        instruments = await client.get_instruments_async(market="power", product="lmp")
        print(f"Found {len(instruments)} LMP instruments")

        if instruments:
            print(f"First 3: {[inst.instrument_id for inst in instruments[:3]]}")

        # Async prices call
        print("\n📈 Getting historical prices asynchronously...")
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=3)

        prices = await client.get_prices_async(
            instrument_id="MISO.HUB.INDIANA",
            start_time=start_time,
            end_time=end_time
        )
        print(f"Retrieved {len(prices)} price ticks asynchronously")

        # Async forward curve call
        print("\n📊 Getting forward curve asynchronously...")
        curve = await client.get_forward_curve_async(
            instrument_id="PJM.HUB.WEST",
            as_of_date=date.today()
        )
        print(f"Retrieved curve with {len(curve.points)} points")

    finally:
        client.close()


async def demo_enhanced_streaming():
    """Demo enhanced WebSocket streaming with async generators."""
    print("\n🌊 Enhanced Real-time Streaming Demo")

    client = CarbonClient(local_dev=True)

    try:
        instruments = ["MISO.HUB.INDIANA", "PJM.HUB.WEST"]
        print(f"📡 Subscribing to: {', '.join(instruments)}")

        # Use async generator for streaming
        price_count = 0
        start_time = datetime.now()

        async for price_tick in client.stream_prices_async(
            instruments,
            reconnect=True,
            max_reconnect_attempts=3
        ):
            price_count += 1
            elapsed = (datetime.now() - start_time).total_seconds()

            print(f"💰 [{elapsed:.1f}s] {price_tick.instrument_id}: ${price_tick.value:.2f}")

            # Stop after 10 updates for demo
            if price_count >= 10:
                break

        print(f"✅ Received {price_count} price updates in {elapsed:.1f} seconds")

    except Exception as e:
        print(f"❌ Streaming error: {e}")
    finally:
        client.close()


async def demo_concurrent_operations():
    """Demo running multiple async operations concurrently."""
    print("\n⚡ Concurrent Operations Demo")

    client = CarbonClient(local_dev=True)

    try:
        # Run multiple async operations concurrently
        tasks = [
            client.get_instruments_async(market="power"),
            client.get_prices_async(
                "MISO.HUB.INDIANA",
                datetime.utcnow() - timedelta(days=1),
                datetime.utcnow()
            ),
            client.get_forward_curve_async(
                "PJM.HUB.WEST",
                date.today()
            )
        ]

        print("🔄 Running 3 async operations concurrently...")

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        instruments, prices, curve = results

        print(f"✅ Instruments: {len(instruments)} found")
        print(f"✅ Prices: {len(prices)} ticks retrieved")
        print(f"✅ Curve: {len(curve.points)} points retrieved")

        # Demonstrate error handling
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Task {i+1} failed: {result}")

    finally:
        client.close()


async def demo_error_handling():
    """Demo error handling and retry logic."""
    print("\n🛡️ Error Handling & Retry Demo")

    # Test with invalid API key to trigger retry logic
    client = CarbonClient(api_key="invalid_key", local_dev=True)

    try:
        # This should trigger retries and eventually fail gracefully
        instruments = await client.get_instruments_async()
        print(f"Unexpected success: {len(instruments)} instruments")

    except Exception as e:
        print(f"✅ Expected error caught: {type(e).__name__}")
        print(f"   Error message: {str(e)[:100]}...")

    finally:
        client.close()


async def main():
    """Run all async demos."""
    print("=" * 70)
    print("🚀 254Carbon Python SDK - Enhanced Async Features Demo")
    print("=" * 70)

    # Run demos sequentially
    await demo_async_api_calls()
    await demo_enhanced_streaming()
    await demo_concurrent_operations()
    await demo_error_handling()

    print("\n" + "=" * 70)
    print("✅ All async demos completed!")
    print("=" * 70)


if __name__ == "__main__":
    # Run the async demo
    asyncio.run(main())
