import { useEffect, useState } from 'react';
import { useEnhancedWebSocketStore } from '../stores/enhancedWebSocketStore';

export function useConnectionStatus() {
  const state = useEnhancedWebSocketStore(s => s.state);
  const metrics = useEnhancedWebSocketStore(s => s.metrics);
  const [connectedAt, setConnectedAt] = useState<number | undefined>(metrics.lastConnectedAt);

  useEffect(() => {
    setConnectedAt(metrics.lastConnectedAt);
  }, [metrics.lastConnectedAt]);

  return { state, metrics, connectedAt } as const;
}


