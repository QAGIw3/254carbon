# 254Carbon Excel Add-in

Real-time market data feeds and analytics functions for Microsoft Excel.

## Features

- **Real-Time Data (RTD)**: Live price updates in Excel cells via WebSocket streaming
- **Custom Functions (UDFs)**: Formula-based data access
- **SSO Authentication**: Integrated with 254Carbon platform
- **Market Coverage**: Power, gas, environmental products

## Installation

### Requirements
- Microsoft Excel 2016 or later (Windows)
- .NET 6.0 Runtime or later
- 254Carbon API key

### Development Setup (Local)

1. **Start the 254Carbon platform:**
   ```bash
   cd platform
   ./scripts/dev-setup.sh
   ```

2. **Build the Excel Add-in:**
   ```bash
   cd apps/excel-addin
   dotnet build
   ```

3. **Register the Add-in:**
   ```bash
   # Build and register (Windows)
   dotnet run -- register
   ```

4. **Load in Excel:**
   - Open Excel
   - Go to File → Options → Add-ins
   - Click "Go" next to "Manage: Excel Add-ins"
   - Click "Browse" and select `254Carbon\bin\Debug\net6.0-windows\254Carbon-AddIn.xll`
   - Check the box next to "254Carbon" and click OK

### Production Installation

1. Download `254Carbon-ExcelAddin.xll` from releases
2. In Excel, go to File → Options → Add-ins
3. Click "Go" next to "Manage: Excel Add-ins"
4. Click "Browse" and select the downloaded `.xll` file
5. Check the box next to "254Carbon" and click OK

## Usage

### Connect to API

#### Local Development
```excel
=C254_CONNECT("dev-key")
```

Or set environment variables:
```bash
export CARBON254_LOCAL_DEV=true
export CARBON254_API_URL=http://localhost:8000
export CARBON254_API_KEY=dev-key
# Optional: override WebSocket endpoint
export CARBON254_WS_URL=ws://localhost:8000/api/v1/stream
```

#### Production
```excel
=C254_CONNECT("your_api_key_here")
```

Or set environment variable:
```bash
export CARBON254_API_KEY=your_api_key
# Optional: override WebSocket endpoint
export CARBON254_WS_URL=wss://api.254carbon.ai/api/v1/stream
```

### Get Live Price (RTD)

```excel
=RTD("Carbon254.RTDServer",,"PRICE","MISO.HUB.INDIANA")
```

The RTD server maintains a persistent WebSocket connection to `/api/v1/stream` for sub-second updates and falls back to REST polling if the socket is unavailable.

### Get Current Price (UDF)

```excel
=C254_PRICE("PJM.HUB.WEST")
```

The price UDF reads from the same in-memory cache that the RTD server maintains. When the WebSocket pushes a new tick, both RTD cells and UDF formulas refresh immediately without an extra API round-trip.

### Get Forward Curve Point

```excel
=C254_CURVE("MISO.HUB.INDIANA", "2026-01", "BASE")
```

Parameters:
- Instrument ID
- Delivery month (YYYY-MM)
- Scenario ID (optional, default: "BASE")

### Get Historical Average

```excel
=C254_HISTORICAL_AVG("PJM.HUB.WEST", 30)
```

30-day average price.

### Calculate VaR

```excel
=C254_VAR("PJM.HUB.WEST", 1000, 0.95)
```

Parameters:
- Instrument ID
- Quantity (MW or contracts)
- Confidence level (default: 0.95)

### List Instruments

```excel
=C254_INSTRUMENTS("power")
```

Returns array of instruments. Use Ctrl+Shift+Enter for array formula.

### Get Analytics Metric

```excel
=C254_ANALYTIC("SHARPE","PJM.HUB.WEST")
```

Parameters:
- Metric name (e.g., `VAR`, `SHARPE`, `PNL`)
- Instrument ID
- Optional dimension or scenario code
- Optional numeric parameter (e.g., VAR quantity)
- Optional confidence level (used for VAR)

For `VAR`, `C254_ANALYTIC` delegates to `C254_VAR`. All analytics results are cached locally and update instantly when the WebSocket delivers new analytics events.

## Example Worksheets

### Price Monitor

| Instrument | Live Price | Avg (30d) | Change % |
|-----------|-----------|----------|----------|
| MISO.HUB.INDIANA | =RTD("Carbon254.RTDServer",,"PRICE","MISO.HUB.INDIANA") | =C254_HISTORICAL_AVG("MISO.HUB.INDIANA",30) | =(B2-C2)/C2 |
| PJM.HUB.WEST | =RTD("Carbon254.RTDServer",,"PRICE","PJM.HUB.WEST") | =C254_HISTORICAL_AVG("PJM.HUB.WEST",30) | =(B3-C3)/C3 |

### Forward Curve

| Month | Price | Month | Price |
|-------|-------|-------|-------|
| 2026-01 | =C254_CURVE("PJM.HUB.WEST","2026-01") | 2026-07 | =C254_CURVE("PJM.HUB.WEST","2026-07") |
| 2026-02 | =C254_CURVE("PJM.HUB.WEST","2026-02") | 2026-08 | =C254_CURVE("PJM.HUB.WEST","2026-08") |

### Portfolio VaR

| Position | Quantity | VaR (95%) | VaR (99%) |
|----------|----------|-----------|-----------|
| MISO Hub | 500 | =C254_VAR("MISO.HUB.INDIANA",500,0.95) | =C254_VAR("MISO.HUB.INDIANA",500,0.99) |
| PJM West | -300 | =C254_VAR("PJM.HUB.WEST",-300,0.95) | =C254_VAR("PJM.HUB.WEST",-300,0.99) |

## Local Development Features

### Mock Data Mode
When running in local development mode, the add-in will automatically fall back to generating realistic mock data if the API is unavailable:

- **Real-time prices** update every 5 seconds with realistic market patterns
- **Forward curves** show contango patterns (increasing prices over time)
- **Historical averages** calculated from mock price history
- **Analytics** (VaR, Sharpe, PnL) use realistic volatility assumptions

### Development Benefits
- ✅ **No API key required** for basic testing
- ✅ **Offline development** possible without internet
- ✅ **Realistic test data** for spreadsheet development
- ✅ **Hot-reloading** when code changes (restart Excel to pick up changes)

## Troubleshooting

### #N/A Error
- Check API key is set correctly
- Verify instrument ID exists
- Check network connection to localhost:8000

### #VALUE! Error
- Verify parameter types (dates as text, numbers as numbers)
- Check formula syntax

### RTD Not Updating
- Ensure Excel calculation is set to Automatic
- Check RTD server is registered (rebuild and reload add-in)
- Verify gateway WebSocket (`/api/v1/stream`) is reachable. Set `CARBON254_WS_URL` if running on a non-default host

### Authentication Errors
- For local development: Use `=C254_CONNECT("dev-key")`
- For production: Verify API key is valid
- Check entitlements for instrument/market
- Try reconnecting: `=C254_CONNECT("your_key")`

### Development Issues
- **Add-in not loading**: Rebuild with `dotnet build` and reload in Excel
- **Mock data not working**: Ensure `CARBON254_LOCAL_DEV=true` environment variable is set
- **API connection issues**: Check that the 254Carbon platform is running on localhost:8000

## Performance Tips

1. **Limit RTD Connections**: Each RTD cell creates a connection. Use sparingly.

2. **Use UDFs for Static Data**: For historical or forward curves, use UDF functions instead of RTD.

3. **Refresh Control**: Disable automatic calculation during data entry:
   ```vba
   Application.Calculation = xlCalculationManual
   ' ... your code ...
   Application.Calculate
   Application.Calculation = xlCalculationAutomatic
   ```

4. **Batch Requests**: Use array formulas to get multiple instruments at once.

## API Reference

### RTD Functions

| Function | Parameters | Description |
|----------|-----------|-------------|
| PRICE | instrument_id | Latest price |
| CURVE | instrument_id, month | Forward curve point |
| FORECAST | instrument_id, months_ahead | ML forecast |

### UDF Functions

| Function | Parameters | Returns |
|----------|-----------|---------|
| C254_CONNECT | api_key, [api_url] | Connection status |
| C254_PRICE | instrument_id | Current price |
| C254_CURVE | instrument_id, month, [scenario] | Curve price |
| C254_HISTORICAL_AVG | instrument_id, [days] | Historical average |
| C254_INSTRUMENTS | [market] | Array of instruments |
| C254_VAR | instrument_id, quantity, [confidence] | Value at Risk |
| C254_ANALYTIC | metric, instrument_id, [dimension], [parameter], [confidence] | Analytics metric value |

## Support

- **Documentation**: https://docs.254carbon.ai/excel
- **Email**: excel-support@254carbon.ai
- **Issues**: https://github.com/254carbon/excel-addin/issues

## License

Proprietary - 254Carbon, Inc.
