import { useEffect, useMemo, useRef, useState } from 'react';
import { useWebSocketStore, PriceUpdate } from '../stores/websocketStore';

export function useRealtimePrices(instrumentIds: string[]) {
  const connect = useWebSocketStore(s => s.connect);
  const disconnect = useWebSocketStore(s => s.disconnect);
  const getLatestPrice = useWebSocketStore(s => s.getLatestPrice);
  const addPriceListener = useWebSocketStore(s => s.addPriceListener);

  const [latest, setLatest] = useState<Record<string, PriceUpdate | null>>({});
  const cleanupRef = useRef<() => void>();

  useEffect(() => {
    connect(instrumentIds);
    cleanupRef.current = addPriceListener((p) => {
      if (instrumentIds.includes(p.instrument_id)) {
        setLatest(prev => ({ ...prev, [p.instrument_id]: p }));
      }
    });
    return () => {
      cleanupRef.current?.();
      disconnect();
    };
  }, [instrumentIds]);

  const latestMap = useMemo(() => {
    const map: Record<string, PriceUpdate | null> = {};
    instrumentIds.forEach(id => { map[id] = getLatestPrice(id); });
    return { ...map, ...latest };
  }, [instrumentIds, latest]);

  return latestMap;
}


