#!/usr/bin/env python3
"""
Example: Advanced analytics with 254Carbon Python SDK

Includes:
- Portfolio VaR (dev mock fallback)
- Correlation matrix
- 30-day price forecast with chart output

Requires matplotlib for plotting: `pip install matplotlib`
"""

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from carbon254 import CarbonClient

def main():
    """Demo advanced analytics features."""
    print("📊 Advanced Analytics Demo")

    # Initialize client for local development
    client = CarbonClient(local_dev=True)

    # Sample portfolio positions
    portfolio = [
        {"instrument_id": "MISO.HUB.INDIANA", "quantity": 1000},
        {"instrument_id": "PJM.HUB.WEST", "quantity": -500},
        {"instrument_id": "CAISO.SP15", "quantity": 750}
    ]

    print("💼 Portfolio VaR Analysis:")

    # Calculate 95% VaR
    var_result = client.get_portfolio_var(portfolio, confidence_level=0.95)
    print(f"   Total VaR (95%): ${var_result['total_var']:.2f}")
    print(f"   Method: {var_result['method']}")

    for position in var_result['positions']:
        print(f"   {position['instrument_id']}: ${position['var_value']:.2f}")

    print("\n📈 Correlation Analysis:")

    # Get correlation matrix for portfolio instruments
    instruments = [p['instrument_id'] for p in portfolio]
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()

    corr_matrix = client.get_correlation_matrix(instruments, start_date, end_date)

    print("   Correlation Matrix:")
    print(corr_matrix.round(3))

    print("\n🔮 Price Forecasting:")

    # Get 30-day forecast for MISO
    forecast = client.get_price_forecast("MISO.HUB.INDIANA", horizon_days=30)

    print(f"   30-day forecast for MISO.HUB.INDIANA:")
    print(f"   Current: ${forecast.iloc[0]['forecast_price']:.2f}")
    print(f"   7-day avg: ${forecast.head(7)['forecast_price'].mean():.2f}")
    print(f"   30-day avg: ${forecast['forecast_price'].mean():.2f}")

    # Plot forecast
    plt.figure(figsize=(12, 6))
    plt.plot(forecast['timestamp'], forecast['forecast_price'], label='Forecast')
    plt.fill_between(forecast['timestamp'], forecast['lower_bound'], forecast['upper_bound'],
                     alpha=0.3, label='80% Confidence Interval')
    plt.title('MISO HUB INDIANA - 30 Day Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price ($/MWh)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('forecast_demo.png', dpi=150, bbox_inches='tight')
    print("   📊 Forecast chart saved as 'forecast_demo.png'")

    client.close()
    print("✅ Analytics demo completed!")

if __name__ == "__main__":
    main()
