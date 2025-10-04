import { useEffect, useState } from 'react';
import { useWebSocketStore, PriceUpdate } from '../stores/websocketStore';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';

interface PriceTickerProps {
  instrumentIds?: string[];
  showHistory?: boolean;
  maxHistoryItems?: number;
}

export default function PriceTicker({
  instrumentIds = ['MISO.HUB.INDIANA', 'PJM.HUB.WEST', 'CAISO.SP15'],
  showHistory = false,
  maxHistoryItems = 10
}: PriceTickerProps) {
  const {
    isConnected,
    connectionError,
    latestPrices,
    getPriceHistory,
    subscribeToPrices
  } = useWebSocketStore();

  const [displayedPrices, setDisplayedPrices] = useState<Record<string, PriceUpdate>>({});

  useEffect(() => {
    // Subscribe to price updates for the specified instruments
    subscribeToPrices(instrumentIds);

    // Update displayed prices when latest prices change
    const relevantPrices: Record<string, PriceUpdate> = {};
    instrumentIds.forEach(id => {
      if (latestPrices[id]) {
        relevantPrices[id] = latestPrices[id];
      }
    });
    setDisplayedPrices(relevantPrices);
  }, [latestPrices, instrumentIds, subscribeToPrices]);

  const formatPrice = (price: number) => `$${price.toFixed(2)}`;

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const getPriceChange = (instrumentId: string) => {
    const history = getPriceHistory(instrumentId, 2);
    if (history.length >= 2) {
      const current = history[history.length - 1].value;
      const previous = history[history.length - 2].value;
      const change = current - previous;
      const percentChange = (change / previous) * 100;

      return {
        change,
        percentChange,
        isPositive: change >= 0
      };
    }
    return null;
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          Real-time Price Ticker
          <Badge
            variant={isConnected ? "default" : "destructive"}
            className="ml-auto"
          >
            {isConnected ? "🟢 Connected" : "🔴 Disconnected"}
          </Badge>
        </CardTitle>
        {connectionError && (
          <p className="text-sm text-red-600">{connectionError}</p>
        )}
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Current Prices */}
        <div className="grid gap-3">
          {instrumentIds.map(instrumentId => {
            const price = displayedPrices[instrumentId];
            const priceChange = getPriceChange(instrumentId);

            return (
              <div
                key={instrumentId}
                className="flex items-center justify-between p-3 border rounded-lg bg-card"
              >
                <div className="flex-1">
                  <div className="font-medium text-sm text-muted-foreground">
                    {instrumentId}
                  </div>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-2xl font-bold">
                      {price ? formatPrice(price.value) : '--'}
                    </span>
                    {priceChange && (
                      <Badge
                        variant={priceChange.isPositive ? "default" : "destructive"}
                        className="text-xs"
                      >
                        {priceChange.isPositive ? '+' : ''}
                        {formatPrice(priceChange.change)}
                        ({priceChange.isPositive ? '+' : ''}
                        {priceChange.percentChange.toFixed(1)}%)
                      </Badge>
                    )}
                  </div>
                  {price && (
                    <div className="text-xs text-muted-foreground mt-1">
                      Last update: {formatTime(price.timestamp)}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Price History (if enabled) */}
        {showHistory && (
          <div className="border-t pt-4">
            <h4 className="font-medium mb-3">Recent Price History</h4>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {instrumentIds.map(instrumentId => {
                const history = getPriceHistory(instrumentId, maxHistoryItems);

                return (
                  <div key={instrumentId} className="text-sm">
                    <div className="font-medium text-muted-foreground mb-1">
                      {instrumentId}
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      {history.slice(-5).map((price, index) => (
                        <div key={index} className="flex justify-between">
                          <span>{formatTime(price.timestamp)}</span>
                          <span className="font-mono">{formatPrice(price.value)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Connection Status */}
        <div className="text-xs text-muted-foreground text-center">
          {isConnected
            ? `Streaming ${Object.keys(displayedPrices).length} instruments`
            : 'Attempting to connect...'
          }
        </div>
      </CardContent>
    </Card>
  );
}
