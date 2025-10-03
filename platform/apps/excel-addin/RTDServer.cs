/*
 * 254Carbon Excel Add-in RTD Server
 * Real-time data feeds for Excel
 */

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Net.Http;
using System.Text.Json;
using Microsoft.Office.Interop.Excel;

namespace Carbon254.ExcelAddin
{
    [
        Guid("A1B2C3D4-E5F6-4A5B-8C9D-0E1F2A3B4C5D"),
        ProgId("Carbon254.RTDServer")
    ]
    public class RTDServer : IRtdServer
    {
        private IRTDUpdateEvent _updateEvent;
        private Timer _updateTimer;
        private readonly Dictionary<int, TopicInfo> _topics = new();
        private readonly HttpClient _httpClient = new();
        private string _apiKey;
        private string _apiBaseUrl = "http://localhost:8000"; // Local development URL
        private bool _useLocalDev = true;

        // Topic info
        class TopicInfo
        {
            public int TopicId { get; set; }
            public string Function { get; set; }
            public string[] Parameters { get; set; }
            public object CurrentValue { get; set; }
        }

        public int ServerStart(IRTDUpdateEvent CallbackObject)
        {
            _updateEvent = CallbackObject;
            
            // Load API configuration
            LoadConfiguration();
            
            // Start update timer (every 5 seconds)
            _updateTimer = new Timer(
                UpdateData,
                null,
                TimeSpan.Zero,
                TimeSpan.FromSeconds(5)
            );
            
            return 1; // Success
        }

        public void ServerTerminate()
        {
            _updateTimer?.Dispose();
            _httpClient?.Dispose();
        }

        public object ConnectData(
            int TopicID,
            ref Array Strings,
            ref bool GetNewValues
        )
        {
            if (Strings.Length == 0)
                return "Invalid parameters";

            string function = Strings.GetValue(0)?.ToString() ?? "";
            var parameters = new string[Strings.Length - 1];
            
            for (int i = 1; i < Strings.Length; i++)
            {
                parameters[i - 1] = Strings.GetValue(i)?.ToString() ?? "";
            }

            var topic = new TopicInfo
            {
                TopicId = TopicID,
                Function = function,
                Parameters = parameters,
                CurrentValue = "Loading..."
            };

            _topics[TopicID] = topic;
            GetNewValues = true;

            // Immediately fetch data
            _ = FetchDataAsync(topic);

            return topic.CurrentValue;
        }

        public void DisconnectData(int TopicID)
        {
            _topics.Remove(TopicID);
        }

        public int Heartbeat()
        {
            return 1;
        }

        public Array RefreshData(ref int TopicCount)
        {
            var updates = new List<object>();
            int count = 0;

            foreach (var topic in _topics.Values)
            {
                updates.Add(topic.TopicId);
                updates.Add(topic.CurrentValue);
                count++;
            }

            TopicCount = count;
            return updates.ToArray();
        }

        // Update all topics
        private async void UpdateData(object state)
        {
            foreach (var topic in _topics.Values)
            {
                await FetchDataAsync(topic);
            }

            if (_topics.Count > 0)
            {
                _updateEvent.UpdateNotify();
            }
        }

        // Fetch data from API
        private async Task FetchDataAsync(TopicInfo topic)
        {
            try
            {
                switch (topic.Function.ToUpper())
                {
                    case "PRICE":
                        await FetchPriceAsync(topic);
                        break;

                    case "CURVE":
                        await FetchCurveAsync(topic);
                        break;

                    case "FORECAST":
                        await FetchForecastAsync(topic);
                        break;

                    default:
                        topic.CurrentValue = $"Unknown function: {topic.Function}";
                        break;
                }
            }
            catch (Exception ex)
            {
                // Try mock data as fallback for development
                if (_useLocalDev)
                {
                    topic.CurrentValue = GenerateMockData(topic);
                }
                else
                {
                    topic.CurrentValue = $"Error: {ex.Message}";
                }
            }
        }

        // Generate mock data for development/testing
        private object GenerateMockData(TopicInfo topic)
        {
            var random = new Random();

            switch (topic.Function.ToUpper())
            {
                case "PRICE":
                    if (topic.Parameters.Length > 0)
                    {
                        string instrumentId = topic.Parameters[0];
                        // Generate realistic price based on instrument
                        if (instrumentId.Contains("MISO"))
                            return 35.0 + random.NextDouble() * 10;
                        else if (instrumentId.Contains("PJM"))
                            return 40.0 + random.NextDouble() * 8;
                        else if (instrumentId.Contains("CAISO"))
                            return 45.0 + random.NextDouble() * 12;
                        else
                            return 3.5 + random.NextDouble() * 1;
                    }
                    return 40.0;

                case "CURVE":
                    if (topic.Parameters.Length >= 2)
                    {
                        string instrumentId = topic.Parameters[0];
                        string month = topic.Parameters[1];

                        // Generate curve price based on delivery month
                        int monthsOut = 1;
                        if (DateTime.TryParse($"2025-{month}-01", out DateTime deliveryDate))
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
                        else
                            return 3.8 + monthsOut * 0.05 + random.NextDouble() * 0.2;
                    }
                    return 42.0;

                case "FORECAST":
                    if (topic.Parameters.Length >= 2)
                    {
                        int monthsAhead = int.Parse(topic.Parameters[1] ?? "1");
                        return 45.0 + monthsAhead * 0.5 + random.NextDouble() * 3;
                    }
                    return 45.0;

                default:
                    return "Mock data unavailable";
            }
        }

        private async Task FetchPriceAsync(TopicInfo topic)
        {
            if (topic.Parameters.Length == 0)
            {
                topic.CurrentValue = "Missing instrument_id";
                return;
            }

            string instrumentId = topic.Parameters[0];
            
            var request = new HttpRequestMessage(
                HttpMethod.Get,
                $"{_apiBaseUrl}/api/v1/prices/latest?instrument_id={instrumentId}"
            );
            
            request.Headers.Add("Authorization", $"Bearer {_apiKey}");
            
            var response = await _httpClient.SendAsync(request);
            
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync();
                var data = JsonSerializer.Deserialize<Dictionary<string, object>>(content);
                
                if (data != null && data.ContainsKey("value"))
                {
                    topic.CurrentValue = data["value"];
                }
            }
            else
            {
                topic.CurrentValue = $"API Error: {response.StatusCode}";
            }
        }

        private async Task FetchCurveAsync(TopicInfo topic)
        {
            if (topic.Parameters.Length < 2)
            {
                topic.CurrentValue = "Missing parameters: instrument_id, month";
                return;
            }

            string instrumentId = topic.Parameters[0];
            string month = topic.Parameters[1];
            
            var request = new HttpRequestMessage(
                HttpMethod.Get,
                $"{_apiBaseUrl}/api/v1/curves/point?instrument_id={instrumentId}&month={month}"
            );
            
            request.Headers.Add("Authorization", $"Bearer {_apiKey}");
            
            var response = await _httpClient.SendAsync(request);
            
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync();
                var data = JsonSerializer.Deserialize<Dictionary<string, object>>(content);
                
                if (data != null && data.ContainsKey("price"))
                {
                    topic.CurrentValue = data["price"];
                }
            }
        }

        private async Task FetchForecastAsync(TopicInfo topic)
        {
            // ML forecast endpoint
            if (topic.Parameters.Length < 2)
            {
                topic.CurrentValue = "Missing parameters";
                return;
            }

            string instrumentId = topic.Parameters[0];
            int monthsAhead = int.Parse(topic.Parameters[1] ?? "1");
            
            // Simplified - would call ML service
            topic.CurrentValue = 45.0 + monthsAhead * 0.5;
        }

        private void LoadConfiguration()
        {
            // Load from Excel Named Range or config file
            // For local development, use localhost by default

            // Check for local development mode
            string localDev = Environment.GetEnvironmentVariable("CARBON254_LOCAL_DEV") ?? "true";
            _useLocalDev = localDev.ToLower() == "true";

            if (_useLocalDev)
            {
                _apiBaseUrl = Environment.GetEnvironmentVariable("CARBON254_API_URL") ?? "http://localhost:8000";
                _apiKey = Environment.GetEnvironmentVariable("CARBON254_API_KEY") ?? "dev-key"; // Default dev key

                // Try to read from Excel named range if available
                try
                {
                    var excelApp = (Application)System.Runtime.InteropServices.Marshal.GetActiveObject("Excel.Application");
                    var workbook = excelApp.ActiveWorkbook;

                    // Look for named ranges
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
                    // Excel not available or named ranges not found - use defaults
                }
            }
            else
            {
                // Production mode
                _apiKey = Environment.GetEnvironmentVariable("CARBON254_API_KEY") ?? "";
                _apiBaseUrl = Environment.GetEnvironmentVariable("CARBON254_API_URL") ?? "https://api.254carbon.ai";
            }
        }
    }
}

