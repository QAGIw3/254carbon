using System;
using System.Collections.Generic;

namespace Carbon254.ExcelAddin
{
    internal static class RealtimeCache
    {
        private sealed class CachedValue
        {
            public object Value { get; init; }
            public DateTime Timestamp { get; init; }
        }

        private static readonly object SyncRoot = new();
        private static readonly Dictionary<string, CachedValue> Store = new();

        private static string BuildKey(string category, params string[] parts)
        {
            return $"{category}::{string.Join("::", parts)}".ToUpperInvariant();
        }

        public static void StorePrice(string instrumentId, object value)
        {
            StoreValue(BuildKey("PRICE", instrumentId), value);
        }

        public static bool TryGetPrice(string instrumentId, TimeSpan ttl, out object value)
        {
            return TryGetValue(BuildKey("PRICE", instrumentId), ttl, out value);
        }

        public static void StoreCurve(string instrumentId, string bucket, object value)
        {
            StoreValue(BuildKey("CURVE", instrumentId, bucket), value);
        }

        public static bool TryGetCurve(string instrumentId, string bucket, TimeSpan ttl, out object value)
        {
            return TryGetValue(BuildKey("CURVE", instrumentId, bucket), ttl, out value);
        }

        public static void StoreAnalytics(string metric, string instrumentId, string dimension, object value)
        {
            StoreValue(BuildKey("ANALYTICS", metric, instrumentId, dimension), value);
        }

        public static bool TryGetAnalytics(string metric, string instrumentId, string dimension, TimeSpan ttl, out object value)
        {
            return TryGetValue(BuildKey("ANALYTICS", metric, instrumentId, dimension), ttl, out value);
        }

        private static void StoreValue(string key, object value)
        {
            lock (SyncRoot)
            {
                Store[key] = new CachedValue
                {
                    Value = value,
                    Timestamp = DateTime.UtcNow
                };
            }
        }

        private static bool TryGetValue(string key, TimeSpan ttl, out object value)
        {
            lock (SyncRoot)
            {
                if (Store.TryGetValue(key, out var cached))
                {
                    if (ttl <= TimeSpan.Zero || DateTime.UtcNow - cached.Timestamp <= ttl)
                    {
                        value = cached.Value;
                        return true;
                    }

                    Store.Remove(key);
                }
            }

            value = null;
            return false;
        }
    }
}
