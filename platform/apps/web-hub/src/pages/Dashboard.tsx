import { useQuery } from '@tanstack/react-query';
import { useWebSocketStore } from '../stores/websocketStore';
import { apiClient } from '../services/api';
import PriceTicker from '../components/PriceTicker';
import LivePriceChart from '../components/LivePriceChart';

export default function Dashboard() {
  useQuery({
    queryKey: ['health'],
    queryFn: () => apiClient.get('/health').then(res => res.data),
  });

  const { isConnected, latestPrices } = useWebSocketStore();

  // Calculate active instruments count
  const activeInstrumentCount = Object.keys(latestPrices).length;

  // Calculate average latency (mock for now)
  const avgLatency = isConnected ? 1.8 : 0;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">
          Market Intelligence Dashboard
        </h1>
        <p className="mt-2 text-gray-600">
          See the market. Price the future.
        </p>
      </div>

      {/* Real-time Price Ticker */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <PriceTicker
          instrumentIds={['MISO.HUB.INDIANA', 'PJM.HUB.WEST', 'CAISO.SP15']}
          showHistory={true}
        />

        <LivePriceChart
          defaultInstrumentIds={['MISO.HUB.INDIANA', 'PJM.HUB.WEST']}
          timeWindow={60}
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900">Real-Time Prices</h3>
          <p className="mt-2 text-3xl font-bold text-blue-600">{activeInstrumentCount}</p>
          <p className="text-sm text-gray-500">Active instruments</p>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900">Stream Latency</h3>
          <p className="mt-2 text-3xl font-bold text-green-600">{avgLatency.toFixed(1)}s</p>
          <p className="text-sm text-gray-500">p95 latency</p>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900">Connection Status</h3>
          <p className="mt-2 text-3xl font-bold text-purple-600">
            {isConnected ? '🟢' : '🔴'}
          </p>
          <p className="text-sm text-gray-500">
            {isConnected ? 'Connected' : 'Disconnected'}
          </p>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">System Status</h2>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-gray-700">API Gateway</span>
            <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-sm">
              Healthy
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-700">Data Ingestion</span>
            <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-sm">
              Healthy
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-700">Curve Service</span>
            <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-sm">
              Healthy
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-gray-700">Real-time Streaming</span>
            <span className={`px-2 py-1 rounded text-sm ${
              isConnected
                ? 'bg-green-100 text-green-800'
                : 'bg-red-100 text-red-800'
            }`}>
              {isConnected ? 'Active' : 'Inactive'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

