/*
 * 254Carbon Excel Add-in RTD Server
 * Real-time data feeds for Excel using WebSocket streaming.
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.WebSockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
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
        private Timer _pollingTimer;
        private readonly Dictionary<int, TopicInfo> _topics = new();
        private readonly HttpClient _httpClient = new();

        private readonly object _topicLock = new();
        private readonly SemaphoreSlim _subscriptionLock = new(1, 1);
        private readonly SemaphoreSlim _connectionLock = new(1, 1);

        private ClientWebSocket _webSocket;
        private CancellationTokenSource _webSocketCts;
        private Task _receiveLoopTask;
        private Uri _websocketUri;
        private volatile bool _webSocketConnected;
        private HashSet<string> _currentWebSocketSubscriptions = new(StringComparer.OrdinalIgnoreCase);
        private int _reconnectScheduled;

        private string _apiKey = string.Empty;
        private string _apiBaseUrl = "http://localhost:8000";
        private bool _useLocalDev = true;

        private static readonly TimeSpan PollInterval = TimeSpan.FromSeconds(5);
        private static readonly TimeSpan ReconnectDelay = TimeSpan.FromSeconds(2);

        private class TopicInfo
        {
            public int TopicId { get; init; }
            public string Function { get; init; } = string.Empty;
            public string[] Parameters { get; init; } = Array.Empty<string>();
            public object CurrentValue { get; set; } = "Loading...";

            public string InstrumentId => Parameters.Length > 0 ? Parameters[0] : string.Empty;
            public string SecondaryKey => Parameters.Length > 1 ? Parameters[1] : string.Empty;
            public string TertiaryKey => Parameters.Length > 2 ? Parameters[2] : string.Empty;
        }

        public int ServerStart(IRTDUpdateEvent CallbackObject)
        {
            _updateEvent = CallbackObject;
            LoadConfiguration();

            _pollingTimer = new Timer(
                UpdateData,
                null,
                PollInterval,
                PollInterval
            );

            return 1;
        }

        public void ServerTerminate()
        {
            _pollingTimer?.Dispose();
            CloseWebSocketAsync().GetAwaiter().GetResult();
            _httpClient.Dispose();
        }

        public object ConnectData(int TopicID, ref Array Strings, ref bool GetNewValues)
        {
            if (Strings.Length == 0)
            {
                return "Invalid parameters";
            }

            string function = Strings.GetValue(0)?.ToString() ?? string.Empty;
            string[] parameters = new string[Strings.Length - 1];

            for (int i = 1; i < Strings.Length; i++)
            {
                parameters[i - 1] = Strings.GetValue(i)?.ToString() ?? string.Empty;
            }

            var topic = new TopicInfo
            {
                TopicId = TopicID,
                Function = function,
                Parameters = parameters,
                CurrentValue = "Loading..."
            };

            lock (_topicLock)
            {
                _topics[TopicID] = topic;
            }

            GetNewValues = true;

            switch (function.ToUpperInvariant())
            {
                case "PRICE":
                    Task.Run(async () =>
                    {
                        if (await FetchDataAsync(topic).ConfigureAwait(false))
                        {
                            _updateEvent?.UpdateNotify();
                        }
                    });
                    ScheduleSubscriptionRefresh();
                    break;

                case "CURVE":
                case "FORECAST":
                case "ANALYTICS":
                    Task.Run(async () =>
                    {
                        if (await FetchDataAsync(topic).ConfigureAwait(false))
                        {
                            _updateEvent?.UpdateNotify();
                        }
                    });
                    break;

                default:
                    topic.CurrentValue = $"Unknown function: {function}";
                    break;
            }

            return topic.CurrentValue;
        }

        public void DisconnectData(int TopicID)
        {
            bool removed; 
            lock (_topicLock)
            {
                removed = _topics.Remove(TopicID);
            }

            if (removed)
            {
                ScheduleSubscriptionRefresh();
            }
        }

        public int Heartbeat() => 1;

        public Array RefreshData(ref int TopicCount)
        {
            List<object> updates = new();

            lock (_topicLock)
            {
                foreach (var topic in _topics.Values)
                {
                    updates.Add(topic.TopicId);
                    updates.Add(topic.CurrentValue);
                }
                TopicCount = _topics.Count;
            }

            return updates.ToArray();
        }

        private async void UpdateData(object state)
        {
            List<TopicInfo> snapshot;
            lock (_topicLock)
            {
                snapshot = _topics.Values.ToList();
            }

            bool valueChanged = false;

            foreach (var topic in snapshot)
            {
                if (!ShouldPoll(topic))
                {
                    continue;
                }

                bool changed = await FetchDataAsync(topic).ConfigureAwait(false);
                valueChanged = valueChanged || changed;
            }

            if (valueChanged)
            {
                _updateEvent?.UpdateNotify();
            }
        }

        private bool ShouldPoll(TopicInfo topic)
        {
            return topic.Function.ToUpperInvariant() switch
            {
                "PRICE" => !_webSocketConnected,
                "CURVE" => true,
                "FORECAST" => true,
                "ANALYTICS" => true,
                _ => false
            };
        }

        private async Task<bool> FetchDataAsync(TopicInfo topic)
        {
            switch (topic.Function.ToUpperInvariant())
            {
                case "PRICE":
                    return await FetchPriceAsync(topic).ConfigureAwait(false);

                case "CURVE":
                    return await FetchCurveAsync(topic).ConfigureAwait(false);

                case "FORECAST":
                    return await FetchForecastAsync(topic).ConfigureAwait(false);

                case "ANALYTICS":
                    return await FetchAnalyticsAsync(topic).ConfigureAwait(false);

                default:
                    return false;
            }
        }

        private async Task<bool> FetchPriceAsync(TopicInfo topic)
        {
            if (string.IsNullOrEmpty(topic.InstrumentId))
            {
                topic.CurrentValue = "Missing instrument_id";
                return true;
            }

            try
            {
                var request = new HttpRequestMessage(
                    HttpMethod.Get,
                    $"{_apiBaseUrl}/api/v1/prices/latest?instrument_id={topic.InstrumentId}"
                );

                if (!string.IsNullOrEmpty(_apiKey))
                {
                    request.Headers.Add("Authorization", $"Bearer {_apiKey}");
                }

                var response = await _httpClient.SendAsync(request).ConfigureAwait(false);

                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
                    var data = JsonSerializer.Deserialize<Dictionary<string, object>>(content);

                    if (data != null && data.TryGetValue("value", out var valueObj) &&
                        double.TryParse(valueObj?.ToString(), out double price))
                    {
                        topic.CurrentValue = price;
                        RealtimeCache.StorePrice(topic.InstrumentId, price);
                        return true;
                    }
                }

                topic.CurrentValue = $"API Error: {response.StatusCode}";
                return true;
            }
            catch (Exception ex)
            {
                if (_useLocalDev)
                {
                    topic.CurrentValue = GenerateMockData(topic);
                }
                else
                {
                    topic.CurrentValue = $"Error: {ex.Message}";
                }
                return true;
            }
        }

        private async Task<bool> FetchCurveAsync(TopicInfo topic)
        {
            if (topic.Parameters.Length < 2)
            {
                topic.CurrentValue = "Missing parameters: instrument_id, month";
                return true;
            }

            string instrumentId = topic.InstrumentId;
            string month = topic.SecondaryKey;
            string scenario = topic.Parameters.Length > 2 ? topic.Parameters[2] : "BASE";

            try
            {
                var request = new HttpRequestMessage(
                    HttpMethod.Get,
                    $"{_apiBaseUrl}/api/v1/curves/point?instrument_id={instrumentId}&month={month}&scenario={scenario}"
                );

                if (!string.IsNullOrEmpty(_apiKey))
                {
                    request.Headers.Add("Authorization", $"Bearer {_apiKey}");
                }

                var response = await _httpClient.SendAsync(request).ConfigureAwait(false);

                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
                    var data = JsonSerializer.Deserialize<Dictionary<string, object>>(content);

                    if (data != null && data.TryGetValue("price", out var priceObj) &&
                        double.TryParse(priceObj?.ToString(), out double price))
                    {
                        topic.CurrentValue = price;
                        RealtimeCache.StoreCurve(instrumentId, month, price);
                        return true;
                    }
                }

                topic.CurrentValue = $"API Error: {response.StatusCode}";
                return true;
            }
            catch (Exception ex)
            {
                if (_useLocalDev)
                {
                    topic.CurrentValue = GenerateMockData(topic);
                }
                else
                {
                    topic.CurrentValue = $"Error: {ex.Message}";
                }
                return true;
            }
        }

        private async Task<bool> FetchForecastAsync(TopicInfo topic)
        {
            if (topic.Parameters.Length < 2)
            {
                topic.CurrentValue = "Missing parameters";
                return true;
            }

            string instrumentId = topic.InstrumentId;
            int monthsAhead = int.TryParse(topic.SecondaryKey, out var months) ? months : 1;

            try
            {
                // Placeholder: actual implementation would call ML forecast service
                var forecast = 45.0 + monthsAhead * 0.5;
                topic.CurrentValue = forecast;
                return true;
            }
            catch (Exception ex)
            {
                if (_useLocalDev)
                {
                    topic.CurrentValue = GenerateMockData(topic);
                }
                else
                {
                    topic.CurrentValue = $"Error: {ex.Message}";
                }
                return true;
            }
        }

        private async Task<bool> FetchAnalyticsAsync(TopicInfo topic)
        {
            if (topic.Parameters.Length == 0)
            {
                topic.CurrentValue = "Missing metric";
                return true;
            }

            string metric = topic.Parameters[0].ToUpperInvariant();

            try
            {
                switch (metric)
                {
                    case "VAR":
                        return await FetchVaRAsync(topic).ConfigureAwait(false);

                    default:
                        topic.CurrentValue = $"Unknown metric: {metric}";
                        return true;
                }
            }
            catch (Exception ex)
            {
                if (_useLocalDev)
                {
                    topic.CurrentValue = GenerateMockData(topic);
                }
                else
                {
                    topic.CurrentValue = $"Error: {ex.Message}";
                }
                return true;
            }
        }

        private async Task<bool> FetchVaRAsync(TopicInfo topic)
        {
            if (topic.Parameters.Length < 3)
            {
                topic.CurrentValue = "Missing parameters: metric, instrument_id, quantity";
                return true;
            }

            string instrumentId = topic.Parameters[1];
            if (!double.TryParse(topic.Parameters[2], out double quantity))
            {
                topic.CurrentValue = "Invalid quantity";
                return true;
            }

            double confidence = 0.95;
            if (topic.Parameters.Length > 3 && double.TryParse(topic.Parameters[3], out double confidenceParam))
            {
                confidence = confidenceParam;
            }

            try
            {
                var payload = new
                {
                    positions = new[]
                    {
                        new { instrument_id = instrumentId, quantity = quantity }
                    },
                    confidence_level = confidence,
                    method = "historical"
                };

                var request = new HttpRequestMessage(
                    HttpMethod.Post,
                    $"{_apiBaseUrl}/api/v1/risk/var"
                )
                {
                    Content = new StringContent(
                        JsonSerializer.Serialize(payload),
                        Encoding.UTF8,
                        "application/json"
                    )
                };

                if (!string.IsNullOrEmpty(_apiKey))
                {
                    request.Headers.Add("Authorization", $"Bearer {_apiKey}");
                }

                var response = await _httpClient.SendAsync(request).ConfigureAwait(false);

                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
                    var data = JsonSerializer.Deserialize<Dictionary<string, object>>(content);

                    if (data != null && data.TryGetValue("var_value", out var valueObj) &&
                        double.TryParse(valueObj?.ToString(), out double varValue))
                    {
                        topic.CurrentValue = varValue;
                        RealtimeCache.StoreAnalytics("VAR", instrumentId, string.Empty, varValue);
                        return true;
                    }
                }

                topic.CurrentValue = $"API Error: {response.StatusCode}";
                return true;
            }
            catch (Exception ex)
            {
                if (_useLocalDev)
                {
                    var mockValue = GenerateMockVaR(topic.Parameters[1], quantity, confidence);
                    topic.CurrentValue = mockValue;
                    RealtimeCache.StoreAnalytics("VAR", instrumentId, string.Empty, mockValue);
                }
                else
                {
                    topic.CurrentValue = $"Error: {ex.Message}";
                }
                return true;
            }
        }

        private object GenerateMockData(TopicInfo topic)
        {
            var random = Random.Shared;

            switch (topic.Function.ToUpperInvariant())
            {
                case "PRICE":
                    if (topic.Parameters.Length > 0)
                    {
                        string instrumentId = topic.InstrumentId;
                        if (instrumentId.Contains("MISO", StringComparison.OrdinalIgnoreCase))
                            return 35.0 + random.NextDouble() * 10;
                        if (instrumentId.Contains("PJM", StringComparison.OrdinalIgnoreCase))
                            return 40.0 + random.NextDouble() * 8;
                        if (instrumentId.Contains("CAISO", StringComparison.OrdinalIgnoreCase))
                            return 45.0 + random.NextDouble() * 12;
                        if (instrumentId.Contains("HENRY", StringComparison.OrdinalIgnoreCase))
                            return 3.5 + random.NextDouble() * 1;
                    }
                    return 40.0 + random.NextDouble() * 5;

                case "CURVE":
                    if (topic.Parameters.Length >= 2)
                    {
                        string instrumentId = topic.InstrumentId;
                        string month = topic.SecondaryKey;
                        int monthsOut = 1;
                        if (DateTime.TryParse($"2025-{month}-01", out DateTime deliveryDate))
                        {
                            monthsOut = Math.Max(1, (deliveryDate.Year - DateTime.Now.Year) * 12 + deliveryDate.Month - DateTime.Now.Month);
                        }

                        if (instrumentId.Contains("MISO", StringComparison.OrdinalIgnoreCase))
                            return 38.0 + monthsOut * 0.5 + random.NextDouble() * 2;
                        if (instrumentId.Contains("PJM", StringComparison.OrdinalIgnoreCase))
                            return 42.0 + monthsOut * 0.3 + random.NextDouble() * 1.5;
                        if (instrumentId.Contains("CAISO", StringComparison.OrdinalIgnoreCase))
                            return 48.0 + monthsOut * 0.7 + random.NextDouble() * 2.5;
                        if (instrumentId.Contains("HENRY", StringComparison.OrdinalIgnoreCase))
                            return 3.8 + monthsOut * 0.05 + random.NextDouble() * 0.2;
                    }
                    return 42.0 + random.NextDouble() * 2;

                case "FORECAST":
                    int monthsAhead = topic.Parameters.Length >= 2 && int.TryParse(topic.SecondaryKey, out var months) ? months : 1;
                    return 45.0 + monthsAhead * 0.5 + random.NextDouble() * 3;

                case "ANALYTICS":
                    string metric = topic.Parameters.Length > 0 ? topic.Parameters[0] : "";
                    if (metric.Equals("VAR", StringComparison.OrdinalIgnoreCase))
                    {
                        double quantity = topic.Parameters.Length > 2 && double.TryParse(topic.Parameters[2], out var qty) ? qty : 10;
                        return GenerateMockVaR(topic.Parameters.Length > 1 ? topic.Parameters[1] : "", quantity, 0.95);
                    }
                    return random.NextDouble() * 100;

                default:
                    return "Mock data unavailable";
            }
        }

        private double GenerateMockVaR(string instrumentId, double quantity, double confidenceLevel)
        {
            var random = Random.Shared;

            double baseVolatility;
            if (!string.IsNullOrEmpty(instrumentId) && instrumentId.Contains("MISO", StringComparison.OrdinalIgnoreCase))
                baseVolatility = 0.25;
            else if (!string.IsNullOrEmpty(instrumentId) && instrumentId.Contains("PJM", StringComparison.OrdinalIgnoreCase))
                baseVolatility = 0.22;
            else if (!string.IsNullOrEmpty(instrumentId) && instrumentId.Contains("CAISO", StringComparison.OrdinalIgnoreCase))
                baseVolatility = 0.30;
            else
                baseVolatility = 0.20;

            double confidenceMultiplier = confidenceLevel >= 0.99 ? 1.3 : 1.0;
            double dailyVolatility = baseVolatility / Math.Sqrt(252);
            double positionValue = Math.Abs(quantity) * 50;
            double zScore = confidenceLevel >= 0.99 ? 2.326 : 1.645;

            double baseVar = positionValue * dailyVolatility * zScore * confidenceMultiplier;
            return baseVar * (0.9 + random.NextDouble() * 0.2);
        }

        private void ScheduleSubscriptionRefresh()
        {
            Task.Run(async () =>
            {
                try
                {
                    await UpdateWebSocketSubscriptionAsync().ConfigureAwait(false);
                }
                catch
                {
                    // Swallow exceptions to avoid impacting Excel UI thread
                }
            });
        }

        private async Task UpdateWebSocketSubscriptionAsync()
        {
            var instruments = GetPriceSubscriptions();

            await _subscriptionLock.WaitAsync().ConfigureAwait(false);
            try
            {
                if (instruments.Count == 0)
                {
                    _currentWebSocketSubscriptions.Clear();
                    await CloseWebSocketAsync().ConfigureAwait(false);
                    return;
                }

                if (_webSocketConnected && _currentWebSocketSubscriptions.SetEquals(instruments))
                {
                    return;
                }

                await RestartWebSocketAsync(instruments).ConfigureAwait(false);
            }
            finally
            {
                _subscriptionLock.Release();
            }
        }

        private HashSet<string> GetPriceSubscriptions()
        {
            lock (_topicLock)
            {
                return _topics.Values
                    .Where(t => t.Function.Equals("PRICE", StringComparison.OrdinalIgnoreCase) && !string.IsNullOrEmpty(t.InstrumentId))
                    .Select(t => t.InstrumentId)
                    .ToHashSet(StringComparer.OrdinalIgnoreCase);
            }
        }

        private async Task RestartWebSocketAsync(HashSet<string> instruments)
        {
            await CloseWebSocketAsync().ConfigureAwait(false);
            await ConnectWebSocketAsync(instruments).ConfigureAwait(false);
        }

        private async Task ConnectWebSocketAsync(HashSet<string> instruments)
        {
            if (_websocketUri == null || instruments.Count == 0)
            {
                return;
            }

            await _connectionLock.WaitAsync().ConfigureAwait(false);
            try
            {
                if (_webSocket != null && _webSocket.State == WebSocketState.Open)
                {
                    return;
                }

                _webSocket = new ClientWebSocket();
                _webSocketCts = new CancellationTokenSource();

                try
                {
                    await _webSocket.ConnectAsync(_websocketUri, _webSocketCts.Token).ConfigureAwait(false);
                }
                catch
                {
                    CleanupWebSocketResources();
                    ScheduleReconnect();
                    return;
                }

                _webSocketConnected = _webSocket.State == WebSocketState.Open;
                if (!_webSocketConnected)
                {
                    CleanupWebSocketResources();
                    ScheduleReconnect();
                    return;
                }

                await SendSubscriptionMessageAsync(instruments, _webSocketCts.Token).ConfigureAwait(false);
                _currentWebSocketSubscriptions = new HashSet<string>(instruments, StringComparer.OrdinalIgnoreCase);

                _receiveLoopTask = Task.Run(() => ReceiveLoopAsync(_webSocketCts.Token));
            }
            finally
            {
                _connectionLock.Release();
            }
        }

        private async Task SendSubscriptionMessageAsync(HashSet<string> instruments, CancellationToken token)
        {
            if (_webSocket == null || _webSocket.State != WebSocketState.Open)
            {
                return;
            }

            var payload = new
            {
                type = "subscribe",
                api_key = _apiKey,
                instruments = instruments.ToArray(),
                channels = new[] { "price" }
            };

            string json = JsonSerializer.Serialize(payload);
            var buffer = Encoding.UTF8.GetBytes(json);

            await _webSocket.SendAsync(new ArraySegment<byte>(buffer), WebSocketMessageType.Text, true, token).ConfigureAwait(false);
        }

        private async Task ReceiveLoopAsync(CancellationToken token)
        {
            var buffer = new byte[8192];

            try
            {
                while (!token.IsCancellationRequested && _webSocket != null && _webSocket.State == WebSocketState.Open)
                {
                    var message = await ReceiveMessageAsync(buffer, token).ConfigureAwait(false);
                    if (message == null)
                    {
                        break;
                    }

                    HandleWebSocketMessage(message);
                }
            }
            catch (OperationCanceledException)
            {
                // Expected during shutdown
            }
            catch
            {
                if (!token.IsCancellationRequested)
                {
                    _webSocketConnected = false;
                    ScheduleReconnect();
                }
            }
        }

        private async Task<string?> ReceiveMessageAsync(byte[] buffer, CancellationToken token)
        {
            if (_webSocket == null)
            {
                return null;
            }

            using var ms = new MemoryStream();
            WebSocketReceiveResult result;

            do
            {
                result = await _webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), token).ConfigureAwait(false);

                if (result.MessageType == WebSocketMessageType.Close)
                {
                    _webSocketConnected = false;
                    ScheduleReconnect();
                    return null;
                }

                ms.Write(buffer, 0, result.Count);
            }
            while (!result.EndOfMessage);

            return Encoding.UTF8.GetString(ms.ToArray());
        }

        private void HandleWebSocketMessage(string message)
        {
            try
            {
                using var document = JsonDocument.Parse(message);
                var root = document.RootElement;

                if (!root.TryGetProperty("type", out var typeProp))
                {
                    return;
                }

                string type = typeProp.GetString() ?? string.Empty;

                switch (type.ToLowerInvariant())
                {
                    case "price_update":
                        HandlePriceUpdate(root);
                        break;

                    case "curve_update":
                        HandleCurveUpdate(root);
                        break;

                    case "analytics_update":
                        HandleAnalyticsUpdate(root);
                        break;

                    case "subscribed":
                        _webSocketConnected = true;
                        break;

                    case "error":
                        _webSocketConnected = false;
                        break;
                }
            }
            catch
            {
                // Ignore malformed messages
            }
        }

        private void HandlePriceUpdate(JsonElement message)
        {
            if (!message.TryGetProperty("data", out var data))
            {
                return;
            }

            string instrumentId = data.TryGetProperty("instrument_id", out var instProp) ? instProp.GetString() ?? string.Empty : string.Empty;
            if (string.IsNullOrEmpty(instrumentId))
            {
                return;
            }

            object value = ExtractValue(data, "value");
            if (value == null)
            {
                return;
            }

            bool updated = false;

            lock (_topicLock)
            {
                foreach (var topic in _topics.Values)
                {
                    if (!topic.Function.Equals("PRICE", StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }

                    if (string.Equals(topic.InstrumentId, instrumentId, StringComparison.OrdinalIgnoreCase))
                    {
                        topic.CurrentValue = value;
                        updated = true;
                    }
                }
            }

            RealtimeCache.StorePrice(instrumentId, value);

            if (updated)
            {
                _updateEvent?.UpdateNotify();
            }
        }

        private void HandleCurveUpdate(JsonElement message)
        {
            if (!message.TryGetProperty("data", out var data))
            {
                return;
            }

            string instrumentId = data.TryGetProperty("instrument_id", out var instProp) ? instProp.GetString() ?? string.Empty : string.Empty;
            string bucket = data.TryGetProperty("bucket", out var bucketProp) ? bucketProp.GetString() ?? string.Empty :
                            data.TryGetProperty("month", out var monthProp) ? monthProp.GetString() ?? string.Empty : string.Empty;

            object value = ExtractValue(data, "price") ?? ExtractValue(data, "value");
            if (string.IsNullOrEmpty(instrumentId) || value == null)
            {
                return;
            }

            bool updated = false;

            lock (_topicLock)
            {
                foreach (var topic in _topics.Values)
                {
                    if (!topic.Function.Equals("CURVE", StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }

                    if (string.Equals(topic.InstrumentId, instrumentId, StringComparison.OrdinalIgnoreCase) &&
                        (string.IsNullOrEmpty(bucket) || string.Equals(topic.SecondaryKey, bucket, StringComparison.OrdinalIgnoreCase)))
                    {
                        topic.CurrentValue = value;
                        updated = true;
                    }
                }
            }

            RealtimeCache.StoreCurve(instrumentId, string.IsNullOrEmpty(bucket) ? "" : bucket, value);

            if (updated)
            {
                _updateEvent?.UpdateNotify();
            }
        }

        private void HandleAnalyticsUpdate(JsonElement message)
        {
            if (!message.TryGetProperty("data", out var data))
            {
                return;
            }

            string metric = data.TryGetProperty("metric", out var metricProp) ? metricProp.GetString() ?? string.Empty : string.Empty;
            string instrumentId = data.TryGetProperty("instrument_id", out var instProp) ? instProp.GetString() ?? string.Empty : string.Empty;
            string bucket = data.TryGetProperty("bucket", out var bucketProp) ? bucketProp.GetString() ?? string.Empty : string.Empty;

            object value = ExtractValue(data, "value");
            if (string.IsNullOrEmpty(metric) || value == null)
            {
                return;
            }

            bool updated = false;

            lock (_topicLock)
            {
                foreach (var topic in _topics.Values)
                {
                    if (!topic.Function.Equals("ANALYTICS", StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }

                    string topicMetric = topic.Parameters.Length > 0 ? topic.Parameters[0] : string.Empty;
                    string topicInstrument = topic.Parameters.Length > 1 ? topic.Parameters[1] : string.Empty;
                    string topicBucket = topic.Parameters.Length > 2 ? topic.Parameters[2] : string.Empty;

                    if (!string.Equals(topicMetric, metric, StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }

                    if (!string.IsNullOrEmpty(topicInstrument) && !string.Equals(topicInstrument, instrumentId, StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }

                    if (!string.IsNullOrEmpty(topicBucket) && !string.Equals(topicBucket, bucket, StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }

                    topic.CurrentValue = value;
                    updated = true;
                }
            }

            RealtimeCache.StoreAnalytics(metric, instrumentId, bucket ?? string.Empty, value);

            if (updated)
            {
                _updateEvent?.UpdateNotify();
            }
        }

        private object ExtractValue(JsonElement data, string propertyName)
        {
            if (!data.TryGetProperty(propertyName, out var prop))
            {
                return null;
            }

            return prop.ValueKind switch
            {
                JsonValueKind.Number => prop.GetDouble(),
                JsonValueKind.String => prop.GetString(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.Null => null,
                _ => prop.ToString()
            };
        }

        private async Task CloseWebSocketAsync()
        {
            await _connectionLock.WaitAsync().ConfigureAwait(false);
            try
            {
                if (_webSocket != null)
                {
                    try
                    {
                        if (_webSocket.State is WebSocketState.Open or WebSocketState.CloseReceived)
                        {
                            await _webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Shutdown", CancellationToken.None).ConfigureAwait(false);
                        }
                    }
                    catch
                    {
                        // Ignore errors during close
                    }
                }

                _webSocketCts?.Cancel();
            }
            finally
            {
                _connectionLock.Release();
            }

            if (_receiveLoopTask != null)
            {
                try
                {
                    await _receiveLoopTask.ConfigureAwait(false);
                }
                catch
                {
                    // Ignore receive loop errors during shutdown
                }
            }

            CleanupWebSocketResources();
            _webSocketConnected = false;
        }

        private void CleanupWebSocketResources()
        {
            try
            {
                _webSocket?.Dispose();
            }
            catch
            {
                // Ignore
            }
            _webSocket = null;

            _webSocketCts?.Dispose();
            _webSocketCts = null;

            _receiveLoopTask = null;
        }

        private void ScheduleReconnect()
        {
            if (Interlocked.Exchange(ref _reconnectScheduled, 1) == 1)
            {
                return;
            }

            Task.Run(async () =>
            {
                try
                {
                    await Task.Delay(ReconnectDelay).ConfigureAwait(false);
                    await UpdateWebSocketSubscriptionAsync().ConfigureAwait(false);
                }
                finally
                {
                    Interlocked.Exchange(ref _reconnectScheduled, 0);
                }
            });
        }

        private void LoadConfiguration()
        {
            string localDev = Environment.GetEnvironmentVariable("CARBON254_LOCAL_DEV") ?? "true";
            _useLocalDev = localDev.Equals("true", StringComparison.OrdinalIgnoreCase);

            if (_useLocalDev)
            {
                _apiBaseUrl = Environment.GetEnvironmentVariable("CARBON254_API_URL") ?? "http://localhost:8000";
                _apiKey = Environment.GetEnvironmentVariable("CARBON254_API_KEY") ?? "dev-key";

                try
                {
                    var excelApp = (Application)Marshal.GetActiveObject("Excel.Application");
                    var workbook = excelApp?.ActiveWorkbook;

                    if (workbook != null)
                    {
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
                }
                catch
                {
                    // Excel not available or named ranges missing
                }
            }
            else
            {
                _apiBaseUrl = Environment.GetEnvironmentVariable("CARBON254_API_URL") ?? "https://api.254carbon.ai";
                _apiKey = Environment.GetEnvironmentVariable("CARBON254_API_KEY") ?? string.Empty;
            }

            string wsOverride = Environment.GetEnvironmentVariable("CARBON254_WS_URL") ?? string.Empty;
            if (!string.IsNullOrEmpty(wsOverride) && Uri.TryCreate(wsOverride, UriKind.Absolute, out var explicitUri))
            {
                _websocketUri = explicitUri;
            }
            else
            {
                _websocketUri = BuildWebSocketUri(_apiBaseUrl);
            }
        }

        private static Uri BuildWebSocketUri(string baseUrl)
        {
            if (!Uri.TryCreate(baseUrl, UriKind.Absolute, out var baseUri))
            {
                return null;
            }

            var builder = new UriBuilder(baseUri)
            {
                Scheme = baseUri.Scheme.Equals("https", StringComparison.OrdinalIgnoreCase) ? "wss" : "ws",
                Port = baseUri.IsDefaultPort ? -1 : baseUri.Port,
                Path = CombinePathSegments(baseUri.AbsolutePath, "api/v1/stream")
            };

            return builder.Uri;
        }

        private static string CombinePathSegments(string basePath, string segment)
        {
            if (string.IsNullOrEmpty(basePath) || basePath == "/")
            {
                return "/" + segment.TrimStart('/');
            }

            return $"{basePath.TrimEnd('/')}/{segment.TrimStart('/')}";
        }
    }
}
