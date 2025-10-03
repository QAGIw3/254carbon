import { useState, useEffect, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../services/api';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

interface PriceTick {
  timestamp: string;
  price: number;
  volume?: number;
  instrument_id: string;
  market: string;
  product: string;
}

interface Instrument {
  instrument_id: string;
  market: string;
  product: string;
  location_code: string;
  unit: string;
  currency: string;
}

export default function MarketExplorer() {
  const [selectedMarket, setSelectedMarket] = useState<string>('MISO');
  const [selectedProduct, setSelectedProduct] = useState<string>('lmp');
  const [selectedInstruments, setSelectedInstruments] = useState<string[]>([]);
  const [priceHistory, setPriceHistory] = useState<PriceTick[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  // Fetch available instruments
  const { data: instruments = [] } = useQuery({
    queryKey: ['instruments', selectedMarket, selectedProduct],
    queryFn: async () => {
      const response = await api.get('/api/v1/instruments', {
        params: { market: selectedMarket, product: selectedProduct }
      });
      return response.data;
    },
  });

  // Fetch initial price history
  const { data: initialPrices } = useQuery({
    queryKey: ['prices', selectedInstruments],
    queryFn: async () => {
      if (selectedInstruments.length === 0) return [];
      const response = await api.get('/api/v1/prices/ticks', {
        params: {
          instrument_ids: selectedInstruments.join(','),
          limit: 100
        }
      });
      return response.data;
    },
    enabled: selectedInstruments.length > 0,
  });

  useEffect(() => {
    if (initialPrices) {
      setPriceHistory(initialPrices.map((tick: any) => ({
        timestamp: new Date(tick.event_time_utc).toLocaleTimeString(),
        price: tick.value,
        volume: tick.volume,
        instrument_id: tick.instrument_id,
        market: tick.market,
        product: tick.product,
      })));
    }
  }, [initialPrices]);

  const startStreaming = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    setIsStreaming(true);

    // WebSocket connection for real-time price updates
    const wsUrl = `ws://localhost:8000/api/v1/stream?instruments=${selectedInstruments.join(',')}`;
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      console.log('WebSocket connected');
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'price_tick') {
        const newTick: PriceTick = {
          timestamp: new Date(data.event_time_utc).toLocaleTimeString(),
          price: data.value,
          volume: data.volume,
          instrument_id: data.instrument_id,
          market: data.market,
          product: data.product,
        };

        setPriceHistory(prev => {
          const updated = [...prev, newTick];
          // Keep only last 100 points for performance
          return updated.slice(-100);
        });
      }
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsStreaming(false);
    };

    wsRef.current.onclose = () => {
      console.log('WebSocket closed');
      setIsStreaming(false);
    };
  };

  const stopStreaming = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsStreaming(false);
  };

  const handleInstrumentToggle = (instrumentId: string) => {
    setSelectedInstruments(prev =>
      prev.includes(instrumentId)
        ? prev.filter(id => id !== instrumentId)
        : [...prev, instrumentId]
    );
  };

  const filteredInstruments = instruments.filter((inst: Instrument) =>
    selectedInstruments.length === 0 || selectedInstruments.includes(inst.instrument_id)
  );

  // Prepare chart data grouped by instrument
  const chartData = priceHistory.reduce((acc: any[], tick) => {
    const existingPoint = acc.find(p => p.timestamp === tick.timestamp);
    if (existingPoint) {
      existingPoint[tick.instrument_id] = tick.price;
    } else {
      const newPoint: any = { timestamp: tick.timestamp };
      newPoint[tick.instrument_id] = tick.price;
      acc.push(newPoint);
    }
    return acc;
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">Market Explorer</h1>
        <div className="flex items-center space-x-4">
          <Badge variant={isStreaming ? "default" : "secondary"}>
            {isStreaming ? "🟢 Streaming" : "🔴 Offline"}
          </Badge>
          <Button
            onClick={isStreaming ? stopStreaming : startStreaming}
            disabled={selectedInstruments.length === 0}
            variant={isStreaming ? "destructive" : "default"}
          >
            {isStreaming ? "Stop Streaming" : "Start Streaming"}
          </Button>
        </div>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle>Filters</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Market
              </label>
              <Select value={selectedMarket} onValueChange={setSelectedMarket}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="MISO">MISO</SelectItem>
                  <SelectItem value="CAISO">CAISO</SelectItem>
                  <SelectItem value="PJM">PJM</SelectItem>
                  <SelectItem value="ERCOT">ERCOT</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Product
              </label>
              <Select value={selectedProduct} onValueChange={setSelectedProduct}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="lmp">LMP</SelectItem>
                  <SelectItem value="capacity">Capacity</SelectItem>
                  <SelectItem value="ancillary">Ancillary Services</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Search Instruments
              </label>
              <Input
                placeholder="Filter instruments..."
                className="w-full"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Instruments List */}
        <Card>
          <CardHeader>
            <CardTitle>Available Instruments</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {filteredInstruments.map((instrument: Instrument) => (
                <div
                  key={instrument.instrument_id}
                  className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                    selectedInstruments.includes(instrument.instrument_id)
                      ? 'bg-blue-50 border-blue-200'
                      : 'hover:bg-gray-50'
                  }`}
                  onClick={() => handleInstrumentToggle(instrument.instrument_id)}
                >
                  <div className="font-medium text-sm">
                    {instrument.location_code}
                  </div>
                  <div className="text-xs text-gray-500">
                    {instrument.market} • {instrument.product}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Real-time Chart */}
        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Price History</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="line" className="w-full">
                <TabsList>
                  <TabsTrigger value="line">Line Chart</TabsTrigger>
                  <TabsTrigger value="area">Area Chart</TabsTrigger>
                </TabsList>

                <TabsContent value="line" className="mt-4">
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis />
                      <Tooltip />
                      {selectedInstruments.map((instrumentId, index) => (
                        <Line
                          key={instrumentId}
                          type="monotone"
                          dataKey={instrumentId}
                          stroke={`hsl(${index * 60}, 70%, 50%)`}
                          strokeWidth={2}
                          dot={false}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </TabsContent>

                <TabsContent value="area" className="mt-4">
                  <ResponsiveContainer width="100%" height={400}>
                    <AreaChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis />
                      <Tooltip />
                      {selectedInstruments.map((instrumentId, index) => (
                        <Area
                          key={instrumentId}
                          type="monotone"
                          dataKey={instrumentId}
                          stackId="1"
                          stroke={`hsl(${index * 60}, 70%, 50%)`}
                          fill={`hsl(${index * 60}, 70%, 50%)`}
                          fillOpacity={0.6}
                        />
                      ))}
                    </AreaChart>
                  </ResponsiveContainer>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Real-time Price Table */}
      {selectedInstruments.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Latest Prices</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2">Instrument</th>
                    <th className="text-right py-2">Latest Price</th>
                    <th className="text-right py-2">Change</th>
                    <th className="text-right py-2">Volume</th>
                    <th className="text-right py-2">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedInstruments.map(instrumentId => {
                    const latestTick = priceHistory
                      .filter(tick => tick.instrument_id === instrumentId)
                      .slice(-1)[0];

                    const previousTick = priceHistory
                      .filter(tick => tick.instrument_id === instrumentId)
                      .slice(-2, -1)[0];

                    const change = latestTick && previousTick
                      ? ((latestTick.price - previousTick.price) / previousTick.price * 100).toFixed(2)
                      : '0.00';

                    return (
                      <tr key={instrumentId} className="border-b hover:bg-gray-50">
                        <td className="py-2 font-medium">
                          {instruments.find(i => i.instrument_id === instrumentId)?.location_code || instrumentId}
                        </td>
                        <td className="text-right py-2 font-mono">
                          {latestTick ? `$${latestTick.price.toFixed(2)}` : '—'}
                        </td>
                        <td className={`text-right py-2 ${Number(change) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {change !== '0.00' ? `${change}%` : '—'}
                        </td>
                        <td className="text-right py-2 font-mono">
                          {latestTick?.volume ? latestTick.volume.toLocaleString() : '—'}
                        </td>
                        <td className="text-right py-2 text-gray-500">
                          {latestTick?.timestamp || '—'}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

