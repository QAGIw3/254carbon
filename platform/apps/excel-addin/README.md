# 254Carbon Excel Add-in

Real-time market data feeds and analytics functions for Microsoft Excel.

## Features

- **Real-Time Data (RTD)**: Live price updates in Excel cells
- **Custom Functions (UDFs)**: Formula-based data access
- **SSO Authentication**: Integrated with 254Carbon platform
- **Market Coverage**: Power, gas, environmental products

## Installation

### Requirements
- Microsoft Excel 2016 or later (Windows)
- .NET Framework 4.7.2 or later
- 254Carbon API key

### Install Steps

1. Download `254Carbon-ExcelAddin.xll` from releases
2. In Excel, go to File → Options → Add-ins
3. Click "Go" next to "Manage: Excel Add-ins"
4. Click "Browse" and select the downloaded `.xll` file
5. Check the box next to "254Carbon" and click OK

## Usage

### Connect to API

```excel
=C254_CONNECT("your_api_key_here")
```

Or set environment variable:
```
CARBON254_API_KEY=your_api_key
```

### Get Live Price (RTD)

```excel
=RTD("Carbon254.RTDServer",,"PRICE","MISO.HUB.INDIANA")
```

Updates automatically every 5 seconds.

### Get Current Price (UDF)

```excel
=C254_PRICE("PJM.HUB.WEST")
```

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

## Troubleshooting

### #N/A Error
- Check API key is set correctly
- Verify instrument ID exists
- Check network connection

### #VALUE! Error
- Verify parameter types (dates as text, numbers as numbers)
- Check formula syntax

### RTD Not Updating
- Ensure Excel calculation is set to Automatic
- Check RTD server is registered: `regsvr32 Carbon254.RTDServer.dll`

### Authentication Errors
- Verify API key is valid
- Check entitlements for instrument/market
- Try reconnecting: `=C254_CONNECT("your_key")`

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

## Support

- **Documentation**: https://docs.254carbon.ai/excel
- **Email**: excel-support@254carbon.ai
- **Issues**: https://github.com/254carbon/excel-addin/issues

## License

Proprietary - 254Carbon, Inc.

