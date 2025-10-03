/*
 * 254Carbon Excel UDF Functions
 * Custom formulas for market data
 */

using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading.Tasks;
using System.Text.Json;
using ExcelDna.Integration;
using Microsoft.Office.Interop.Excel;

namespace Carbon254.ExcelAddin
{
    public class Functions
    {
        private static readonly HttpClient _httpClient = new();
        private static string _apiKey = "";
        private static string _apiBaseUrl = "http://localhost:8000"; // Local development URL
        private static bool _useLocalDev = true;
        private static readonly TimeSpan PriceCacheTtl = TimeSpan.FromSeconds(5);
        private static readonly TimeSpan CurveCacheTtl = TimeSpan.FromSeconds(30);
        private static readonly TimeSpan AnalyticsCacheTtl = TimeSpan.FromSeconds(10);

        // Initialize connection
        [ExcelFunction(Description = "Set 254Carbon API credentials")]
        public static string C254_CONNECT(string apiKey, string apiUrl = "")
        {
            _apiKey = apiKey;

            // Check for local development mode
            string localDev = Environment.GetEnvironmentVariable("CARBON254_LOCAL_DEV") ?? "true";
            _useLocalDev = localDev.ToLower() == "true";

            if (_useLocalDev)
            {
                _apiBaseUrl = !string.IsNullOrEmpty(apiUrl) ? apiUrl : "http://localhost:8000";
                _apiKey = string.IsNullOrEmpty(apiKey) ? "dev-key" : apiKey;

                // Try to read from Excel named range if no explicit key provided
                if (string.IsNullOrEmpty(apiKey) || apiKey == "dev-key")
                {
                    try
                    {
                        var excelApp = (Application)System.Runtime.InteropServices.Marshal.GetActiveObject("Excel.Application");
                        var workbook = excelApp.ActiveWorkbook;

                        foreach (Name name in workbook.Names)
                        {
                            if (name.Name == "CarbonAPIKey")
                            {
                                _apiKey = name.RefersToRange.Value?.ToString() ?? _apiKey;
                            }
                            if (name.Name == "CarbonAPIUrl")
                            {
                                _apiBaseUrl = name.RefersToRange.Value?.ToString() ?? _apiBaseUrl;
                            }
                        }
                    }
                    catch
                    {
                        // Excel not available - use defaults
                    }
                }
            }
            else
            {
                // Production mode
                _apiBaseUrl = !string.IsNullOrEmpty(apiUrl) ? apiUrl : "https://api.254carbon.ai";
            }

            Environment.SetEnvironmentVariable("CARBON254_API_KEY", _apiKey);
            Environment.SetEnvironmentVariable("CARBON254_API_URL", _apiBaseUrl);

            return $"Connected to 254Carbon ({(_useLocalDev ? "Local Dev" : "Production")})";
        }

        // Get current price
        [ExcelFunction(Description = "Get latest price for instrument")]
        public static object C254_PRICE(string instrumentId)
        {
            if (RealtimeCache.TryGetPrice(instrumentId, PriceCacheTtl, out var cachedValue))
            {
                return cachedValue;
            }

            return ExcelAsyncUtil.Run("C254_PRICE", new object[] { instrumentId }, async () =>
            {
                try
                {
                    var request = new HttpRequestMessage(
                        HttpMethod.Get,
                        $"{_apiBaseUrl}/api/v1/prices/latest?instrument_id={instrumentId}"
                    );
                    
                    request.Headers.Add("Authorization", $"Bearer {_apiKey}");
                    
                    var response = await _httpClient.SendAsync(request);
                    var content = await response.Content.ReadAsStringAsync();
                    var data = JsonSerializer.Deserialize<Dictionary<string, object>>(content);
                    
                    if (data != null && data.ContainsKey("value"))
                    {
                        double price = Convert.ToDouble(data["value"].ToString());
                        RealtimeCache.StorePrice(instrumentId, price);
                        return price;
                    }

                    return ExcelError.ExcelErrorNA;
                }
                catch (Exception ex)
                {
                    // Try mock data as fallback for development
                    if (_useLocalDev)
                    {
                        var mockPrice = GenerateMockPrice(instrumentId);
                        RealtimeCache.StorePrice(instrumentId, mockPrice);
                        return mockPrice;
                    }
                    return $"Error: {ex.Message}";
                }
            });
        }

        // Generate mock data for development/testing
        private static double GenerateMockPrice(string instrumentId)
        {
            var random = Random.Shared;

            // Generate realistic price based on instrument
            if (instrumentId.Contains("MISO"))
                return 35.0 + random.NextDouble() * 10;
            else if (instrumentId.Contains("PJM"))
                return 40.0 + random.NextDouble() * 8;
            else if (instrumentId.Contains("CAISO"))
                return 45.0 + random.NextDouble() * 12;
            else if (instrumentId.Contains("HENRY"))
                return 3.5 + random.NextDouble() * 1;
            else
                return 40.0 + random.NextDouble() * 5;
        }

        // Generate mock curve data for development/testing
        private static double GenerateMockCurve(string instrumentId, string deliveryMonth)
        {
            var random = Random.Shared;

            // Generate curve price based on delivery month
            int monthsOut = 1;
            if (DateTime.TryParse($"2025-{deliveryMonth}-01", out DateTime deliveryDate))
            {
                monthsOut = Math.Max(1, (deliveryDate.Year - DateTime.Now.Year) * 12 + deliveryDate.Month - DateTime.Now.Month);
            }

            // Contango curve - prices increase with time
            if (instrumentId.Contains("MISO"))
                return 38.0 + monthsOut * 0.5 + random.NextDouble() * 2;
            else if (instrumentId.Contains("PJM"))
                return 42.0 + monthsOut * 0.3 + random.NextDouble() * 1.5;
            else if (instrumentId.Contains("CAISO"))
                return 48.0 + monthsOut * 0.7 + random.NextDouble() * 2.5;
            else if (instrumentId.Contains("HENRY"))
                return 3.8 + monthsOut * 0.05 + random.NextDouble() * 0.2;
            else
                return 42.0 + monthsOut * 0.4 + random.NextDouble() * 1.8;
        }

        // Generate mock historical average data for development/testing
        private static double GenerateMockHistoricalAvg(string instrumentId, int days)
        {
            var random = Random.Shared;

            // Generate average based on instrument and time period
            double basePrice = GenerateMockPrice(instrumentId);
            double volatility = basePrice * 0.15; // 15% volatility

            // Add some trend based on days (longer periods have more stable averages)
            double trendFactor = Math.Min(1.0, days / 90.0); // Stabilize over 90 days
            return basePrice + (random.NextDouble() - 0.5) * volatility * (1.0 - trendFactor);
        }

        // Get forward curve point
        [ExcelFunction(Description = "Get forward curve price for specific month")]
        public static object C254_CURVE(
            string instrumentId,
            string deliveryMonth,
            string scenario = "BASE"
        )
        {
            if (RealtimeCache.TryGetCurve(instrumentId, deliveryMonth, CurveCacheTtl, out var cachedValue))
            {
                return cachedValue;
            }

            return ExcelAsyncUtil.Run("C254_CURVE", new object[] { instrumentId, deliveryMonth }, async () =>
            {
                try
                {
                    var request = new HttpRequestMessage(
                        HttpMethod.Get,
                        $"{_apiBaseUrl}/api/v1/curves/point?instrument_id={instrumentId}&month={deliveryMonth}&scenario={scenario}"
                    );
                    
                    request.Headers.Add("Authorization", $"Bearer {_apiKey}");
                    
                    var response = await _httpClient.SendAsync(request);
                    var content = await response.Content.ReadAsStringAsync();
                    var data = JsonSerializer.Deserialize<Dictionary<string, object>>(content);
                    
                    if (data != null && data.ContainsKey("price"))
                    {
                        double curvePrice = Convert.ToDouble(data["price"].ToString());
                        RealtimeCache.StoreCurve(instrumentId, deliveryMonth, curvePrice);
                        return curvePrice;
                    }

                    return ExcelError.ExcelErrorNA;
                }
                catch (Exception ex)
                {
                    // Try mock data as fallback for development
                    if (_useLocalDev)
                    {
                        var mockCurve = GenerateMockCurve(instrumentId, deliveryMonth);
                        RealtimeCache.StoreCurve(instrumentId, deliveryMonth, mockCurve);
                        return mockCurve;
                    }
                    return $"Error: {ex.Message}";
                }
            });
        }

        // Get historical average
        [ExcelFunction(Description = "Get historical price average")]
        public static object C254_HISTORICAL_AVG(
            string instrumentId,
            int days = 30
        )
        {
            return ExcelAsyncUtil.Run("C254_HIST_AVG", new object[] { instrumentId, days }, async () =>
            {
                try
                {
                    var endDate = DateTime.UtcNow;
                    var startDate = endDate.AddDays(-days);
                    
                    var request = new HttpRequestMessage(
                        HttpMethod.Get,
                        $"{_apiBaseUrl}/api/v1/prices/average?instrument_id={instrumentId}&start={startDate:yyyy-MM-dd}&end={endDate:yyyy-MM-dd}"
                    );
                    
                    request.Headers.Add("Authorization", $"Bearer {_apiKey}");
                    
                    var response = await _httpClient.SendAsync(request);
                    var content = await response.Content.ReadAsStringAsync();
                    var data = JsonSerializer.Deserialize<Dictionary<string, object>>(content);
                    
                    if (data != null && data.ContainsKey("average"))
                    {
                        return Convert.ToDouble(data["average"].ToString());
                    }

                    return ExcelError.ExcelErrorNA;
                }
                catch (Exception ex)
                {
                    // Try mock data as fallback for development
                    if (_useLocalDev)
                    {
                        return GenerateMockHistoricalAvg(instrumentId, days);
                    }
                    return $"Error: {ex.Message}";
                }
            });
        }

        // Get instruments list
        [ExcelFunction(Description = "Get list of available instruments")]
        public static object[,] C254_INSTRUMENTS(string market = "")
        {
            var task = Task.Run(async () =>
            {
                try
                {
                    var url = $"{_apiBaseUrl}/api/v1/instruments";
                    if (!string.IsNullOrEmpty(market))
                        url += $"?market={market}";
                    
                    var request = new HttpRequestMessage(HttpMethod.Get, url);
                    request.Headers.Add("Authorization", $"Bearer {_apiKey}");
                    
                    var response = await _httpClient.SendAsync(request);
                    var content = await response.Content.ReadAsStringAsync();
                    var instruments = JsonSerializer.Deserialize<List<Dictionary<string, object>>>(content);
                    
                    if (instruments == null || instruments.Count == 0)
                        return new object[,] { { "No instruments found" } };
                    
                    var result = new object[instruments.Count, 3];
                    for (int i = 0; i < instruments.Count; i++)
                    {
                        result[i, 0] = instruments[i]["instrument_id"];
                        result[i, 1] = instruments[i]["market"];
                        result[i, 2] = instruments[i]["location_code"];
                    }
                    
                    return result;
                }
                catch
                {
                    return new object[,] { { "Error fetching instruments" } };
                }
            });

            task.Wait();
            return task.Result;
        }

        // Calculate VaR
        [ExcelFunction(Description = "Calculate Value at Risk for position")]
        public static object C254_VAR(
            string instrumentId,
            double quantity,
            double confidenceLevel = 0.95
        )
        {
            if (RealtimeCache.TryGetAnalytics("VAR", instrumentId, string.Empty, AnalyticsCacheTtl, out var cachedValue))
            {
                return cachedValue;
            }

            return ExcelAsyncUtil.Run("C254_VAR", new object[] { instrumentId, quantity }, async () =>
            {
                try
                {
                    var payload = new
                    {
                        positions = new[]
                        {
                            new { instrument_id = instrumentId, quantity = quantity }
                        },
                        confidence_level = confidenceLevel,
                        method = "historical"
                    };
                    
                    var request = new HttpRequestMessage(
                        HttpMethod.Post,
                        $"{_apiBaseUrl}/api/v1/risk/var"
                    )
                    {
                        Content = new StringContent(
                            JsonSerializer.Serialize(payload),
                            System.Text.Encoding.UTF8,
                            "application/json"
                        )
                    };
                    
                    request.Headers.Add("Authorization", $"Bearer {_apiKey}");
                    
                    var response = await _httpClient.SendAsync(request);
                    var content = await response.Content.ReadAsStringAsync();
                    var data = JsonSerializer.Deserialize<Dictionary<string, object>>(content);
                    
                    if (data != null && data.ContainsKey("var_value"))
                    {
                        double varValue = Convert.ToDouble(data["var_value"].ToString());
                        RealtimeCache.StoreAnalytics("VAR", instrumentId, string.Empty, varValue);
                        return varValue;
                    }

                    return ExcelError.ExcelErrorNA;
                }
                catch (Exception ex)
                {
                    // Try mock data as fallback for development
                    if (_useLocalDev)
                    {
                        var mockVar = GenerateMockVaR(instrumentId, quantity, confidenceLevel);
                        RealtimeCache.StoreAnalytics("VAR", instrumentId, string.Empty, mockVar);
                        return mockVar;
                    }
                    return $"Error: {ex.Message}";
                }
            });
        }

        [ExcelFunction(Description = "Get analytics metric value (e.g., VAR, SHARPE)")]
        public static object C254_ANALYTIC(
            string metric,
            string instrumentId,
            object dimension = null,
            object parameter = null,
            double confidenceLevel = 0.95
        )
        {
            string normalizedMetric = metric?.ToUpperInvariant() ?? string.Empty;
            string instrumentKey = instrumentId ?? string.Empty;
            string dimensionKey = NormalizeOptionalArgument(dimension);

            if (normalizedMetric == "VAR")
            {
                double quantity = ParseOptionalDouble(parameter, 1.0);
                return C254_VAR(instrumentKey, quantity, confidenceLevel);
            }

            if (RealtimeCache.TryGetAnalytics(normalizedMetric, instrumentKey, dimensionKey, AnalyticsCacheTtl, out var cachedValue))
            {
                return cachedValue;
            }

            return ExcelAsyncUtil.Run("C254_ANALYTIC", new object[] { normalizedMetric, instrumentKey, dimensionKey, parameter ?? ExcelMissing.Value, confidenceLevel }, async () =>
            {
                try
                {
                    string url = $"{_apiBaseUrl}/api/v1/analytics/value?metric={Uri.EscapeDataString(normalizedMetric)}&instrument_id={Uri.EscapeDataString(instrumentKey)}";

                    if (!string.IsNullOrEmpty(dimensionKey))
                    {
                        url += $"&dimension={Uri.EscapeDataString(dimensionKey)}";
                    }

                    string parameterValue = NormalizeOptionalArgument(parameter);
                    if (!string.IsNullOrEmpty(parameterValue))
                    {
                        url += $"&parameter={Uri.EscapeDataString(parameterValue)}";
                    }

                    var request = new HttpRequestMessage(HttpMethod.Get, url);
                    request.Headers.Add("Authorization", $"Bearer {_apiKey}");

                    var response = await _httpClient.SendAsync(request);
                    var content = await response.Content.ReadAsStringAsync();

                    if (response.IsSuccessStatusCode)
                    {
                        var data = JsonSerializer.Deserialize<Dictionary<string, object>>(content);
                        if (data != null && data.TryGetValue("value", out var valueObj))
                        {
                            object value = valueObj;
                            if (double.TryParse(valueObj?.ToString(), out double numericValue))
                            {
                                value = numericValue;
                            }

                            RealtimeCache.StoreAnalytics(normalizedMetric, instrumentKey, dimensionKey, value);
                            return value;
                        }
                    }

                    return ExcelError.ExcelErrorNA;
                }
                catch (Exception ex)
                {
                    if (_useLocalDev)
                    {
                        var mockValue = GenerateMockAnalytics(normalizedMetric, instrumentKey, dimensionKey, parameter);
                        RealtimeCache.StoreAnalytics(normalizedMetric, instrumentKey, dimensionKey, mockValue);
                        return mockValue;
                    }

                    return $"Error: {ex.Message}";
                }
            });
        }

        // Generate mock VaR data for development/testing
        private static double GenerateMockVaR(string instrumentId, double quantity, double confidenceLevel)
        {
            var random = Random.Shared;

            // Base volatility by instrument
            double baseVolatility;
            if (instrumentId.Contains("MISO"))
                baseVolatility = 0.25; // 25% annual volatility
            else if (instrumentId.Contains("PJM"))
                baseVolatility = 0.22; // 22% annual volatility
            else if (instrumentId.Contains("CAISO"))
                baseVolatility = 0.30; // 30% annual volatility
            else
                baseVolatility = 0.20; // 20% annual volatility

            // Adjust for confidence level (higher confidence = higher VaR)
            double confidenceMultiplier = confidenceLevel switch
            {
                0.95 => 1.0,
                0.99 => 1.3,
                _ => 1.0
            };

            // Calculate VaR using simplified formula
            double dailyVolatility = baseVolatility / Math.Sqrt(252); // Annual to daily
            double positionValue = Math.Abs(quantity) * 50; // Assume $50/MWh average price
            double zScore = confidenceLevel == 0.99 ? 2.326 : 1.645; // Z-score for confidence level

            return positionValue * dailyVolatility * zScore * confidenceMultiplier;
        }

        private static double GenerateMockAnalytics(string metric, string instrumentId, string dimension, object parameter)
        {
            var random = Random.Shared;

            return metric switch
            {
                "SHARPE" => 1.0 + random.NextDouble() * 0.5,
                "PNL" => (random.NextDouble() - 0.5) * 10000,
                "DRAWDOWN" => -(random.NextDouble() * 5),
                _ => GenerateMockVaR(instrumentId, ParseOptionalDouble(parameter, 10.0), 0.95)
            };
        }

        private static string NormalizeOptionalArgument(object value)
        {
            if (value == null || value is ExcelMissing)
            {
                return string.Empty;
            }

            if (value is double doubleValue)
            {
                return doubleValue.ToString();
            }

            return value.ToString() ?? string.Empty;
        }

        private static double ParseOptionalDouble(object value, double defaultValue)
        {
            if (value == null || value is ExcelMissing)
            {
                return defaultValue;
            }

            if (value is double d)
            {
                return d;
            }

            if (value is string s && double.TryParse(s, out double parsed))
            {
                return parsed;
            }

            return defaultValue;
        }
    }
}
