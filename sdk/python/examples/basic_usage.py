"""
Basic usage examples for 254Carbon Python SDK.

Prerequisites:
- Install: `pip install carbon254` (or run from repo root)
- Set API key: use CarbonClient(api_key=...) or local_dev=True for mocks
"""
from carbon254 import CarbonClient
from datetime import datetime, timedelta, date

def main():
    # Initialize client
    client = CarbonClient(api_key="your_api_key_here")
    
    print("=" * 60)
    print("254Carbon Python SDK - Basic Usage Examples")
    print("=" * 60)
    
    # Example 1: Get instruments
    print("\n1. Getting Power Market Instruments...")
    instruments = client.get_instruments(market="power", product="lmp")
    print(f"Found {len(instruments)} LMP instruments")
    
    if instruments:
        print(f"\nFirst 5 instruments:")
        for inst in instruments[:5]:
            print(f"  - {inst.instrument_id}: {inst.location_code}")
    
    # Example 2: Get historical prices
    print("\n2. Getting Historical Prices...")
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=7)
    
    df = client.get_prices_dataframe(
        instrument_id="MISO.HUB.INDIANA",
        start_time=start_time,
        end_time=end_time,
    )
    
    if not df.empty:
        print(f"\nRetrieved {len(df)} price points")
        print(f"Average price: ${df['price'].mean():.2f}/MWh")
        print(f"Max price: ${df['price'].max():.2f}/MWh")
        print(f"Min price: ${df['price'].min():.2f}/MWh")
        print(f"\nLast 5 prices:")
        print(df[['price', 'volume']].tail())
    
    # Example 3: Get forward curve
    print("\n3. Getting Forward Curve...")
    curve_df = client.get_curve_dataframe(
        instrument_id="PJM.HUB.WEST",
        as_of_date=date.today(),
        scenario_id="BASE",
    )
    
    if not curve_df.empty:
        print(f"\nRetrieved curve with {len(curve_df)} points")
        print(f"First year average: ${curve_df['price'].head(12).mean():.2f}/MWh")
        print(f"\nFirst 6 months:")
        print(curve_df[['price', 'tenor_type']].head(6))
    
    # Example 4: Create and run scenario
    print("\n4. Creating Custom Scenario...")
    try:
        scenario_id = client.create_scenario(
            title="SDK Test Scenario",
            description="Testing scenario creation via Python SDK",
            assumptions={
                "as_of_date": "2025-10-03",
                "power": {
                    "load_growth": {"PJM": 2.0}
                }
            }
        )
        print(f"Created scenario: {scenario_id}")
        
        # Run scenario
        run_id = client.run_scenario(scenario_id)
        print(f"Started run: {run_id}")
        
        # Check status
        import time
        for i in range(5):
            status = client.get_run_status(scenario_id, run_id)
            print(f"Run status: {status['status']}")
            if status['status'] in ['success', 'failed']:
                break
            time.sleep(2)
    except Exception as e:
        print(f"Scenario creation failed: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    
    client.close()


if __name__ == "__main__":
    main()
