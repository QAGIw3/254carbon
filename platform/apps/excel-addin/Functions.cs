/*
 * 254Carbon Excel UDF Functions
 * Custom formulas for market data
 */

using System;
using System.Net.Http;
using System.Threading.Tasks;
using System.Text.Json;
using ExcelDna.Integration;

namespace Carbon254.ExcelAddin
{
    public class Functions
    {
        private static readonly HttpClient _httpClient = new();
        private static string _apiKey = "";
        private static string _apiBaseUrl = "https://api.254carbon.ai";

        // Initialize connection
        [ExcelFunction(Description = "Set 254Carbon API credentials")]
        public static string C254_CONNECT(string apiKey, string apiUrl = "")
        {
            _apiKey = apiKey;
            if (!string.IsNullOrEmpty(apiUrl))
                _apiBaseUrl = apiUrl;
            
            return "Connected to 254Carbon";
        }

        // Get current price
        [ExcelFunction(Description = "Get latest price for instrument")]
        public static object C254_PRICE(string instrumentId)
        {
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
                        return Convert.ToDouble(data["value"].ToString());
                    }
                    
                    return ExcelError.ExcelErrorNA;
                }
                catch (Exception ex)
                {
                    return $"Error: {ex.Message}";
                }
            });
        }

        // Get forward curve point
        [ExcelFunction(Description = "Get forward curve price for specific month")]
        public static object C254_CURVE(
            string instrumentId,
            string deliveryMonth,
            string scenario = "BASE"
        )
        {
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
                        return Convert.ToDouble(data["price"].ToString());
                    }
                    
                    return ExcelError.ExcelErrorNA;
                }
                catch (Exception ex)
                {
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
                        return Convert.ToDouble(data["var_value"].ToString());
                    }
                    
                    return ExcelError.ExcelErrorNA;
                }
                catch (Exception ex)
                {
                    return $"Error: {ex.Message}";
                }
            });
        }
    }
}

