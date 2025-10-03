import { useQuery } from '@tanstack/react-query';
import { api } from '../services/api';

export default function Dashboard() {
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: () => api.get('/health').then(res => res.data),
  });

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

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900">Real-Time Prices</h3>
          <p className="mt-2 text-3xl font-bold text-blue-600">2,847</p>
          <p className="text-sm text-gray-500">Active instruments</p>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900">Stream Latency</h3>
          <p className="mt-2 text-3xl font-bold text-green-600">1.8s</p>
          <p className="text-sm text-gray-500">p95 latency</p>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900">Forecast Scenarios</h3>
          <p className="mt-2 text-3xl font-bold text-purple-600">24</p>
          <p className="text-sm text-gray-500">Active scenarios</p>
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
        </div>
      </div>
    </div>
  );
}

