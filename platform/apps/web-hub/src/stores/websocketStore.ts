import { create } from 'zustand';
import { webSocketManager } from '../services/api';

export interface PriceUpdate {
  instrument_id: string;
  value: number;
  timestamp: string;
  source: string;
  market: string;
  product: string;
}

export interface WebSocketState {
  // Connection state
  isConnected: boolean;
  connectionError: string | null;

  // Real-time data
  latestPrices: Record<string, PriceUpdate>;
  priceHistory: Record<string, PriceUpdate[]>;

  // Connection management
  connect: (instrumentIds?: string[]) => Promise<void>;
  disconnect: () => void;
  subscribeToPrices: (instrumentIds: string[]) => void;

  // Data management
  getLatestPrice: (instrumentId: string) => PriceUpdate | null;
  getPriceHistory: (instrumentId: string, limit?: number) => PriceUpdate[];
  clearPriceHistory: (instrumentId?: string) => void;

  // Listeners for components
  addPriceListener: (callback: (price: PriceUpdate) => void) => () => void;
}

export const useWebSocketStore = create<WebSocketState>((set, get) => ({
  // Initial state
  isConnected: false,
  connectionError: null,
  latestPrices: {},
  priceHistory: {},

  // Connection management
  connect: async (instrumentIds = []) => {
    try {
      await webSocketManager.connect(instrumentIds);
      set({ isConnected: true, connectionError: null });

      // Set up price update listener
      const cleanup = webSocketManager.addListener('price_update', (priceData: PriceUpdate) => {
        set(state => {
          const newLatestPrices = { ...state.latestPrices };
          const newPriceHistory = { ...state.priceHistory };

          // Update latest price
          newLatestPrices[priceData.instrument_id] = priceData;

          // Add to price history (keep last 100 updates per instrument)
          if (!newPriceHistory[priceData.instrument_id]) {
            newPriceHistory[priceData.instrument_id] = [];
          }

          newPriceHistory[priceData.instrument_id].push(priceData);
          if (newPriceHistory[priceData.instrument_id].length > 100) {
            newPriceHistory[priceData.instrument_id] = newPriceHistory[priceData.instrument_id].slice(-100);
          }

          return {
            latestPrices: newLatestPrices,
            priceHistory: newPriceHistory,
          };
        });
      });

      // Store cleanup function for disconnection
      (webSocketManager as any)._priceListenerCleanup = cleanup;

    } catch (error) {
      console.error('WebSocket connection failed:', error);
      set({
        isConnected: false,
        connectionError: error instanceof Error ? error.message : 'Connection failed'
      });
    }
  },

  disconnect: () => {
    webSocketManager.disconnect();
    set({
      isConnected: false,
      connectionError: null,
      latestPrices: {},
      priceHistory: {}
    });

    // Clean up listener
    if ((webSocketManager as any)._priceListenerCleanup) {
      (webSocketManager as any)._priceListenerCleanup();
    }
  },

  subscribeToPrices: (instrumentIds: string[]) => {
    if (get().isConnected) {
      // Reconnect with new instrument subscriptions
      get().disconnect();
      setTimeout(() => get().connect(instrumentIds), 100);
    }
  },

  // Data accessors
  getLatestPrice: (instrumentId: string) => {
    return get().latestPrices[instrumentId] || null;
  },

  getPriceHistory: (instrumentId: string, limit = 50) => {
    const history = get().priceHistory[instrumentId] || [];
    return history.slice(-limit);
  },

  clearPriceHistory: (instrumentId?: string) => {
    if (instrumentId) {
      set(state => ({
        priceHistory: {
          ...state.priceHistory,
          [instrumentId]: []
        }
      }));
    } else {
      set({ priceHistory: {} });
    }
  },

  // Listener management for components
  addPriceListener: (callback: (price: PriceUpdate) => void) => {
    return webSocketManager.addListener('price_update', callback);
  },
}));

// Auto-connect when store is first used (for development)
if (typeof window !== 'undefined') {
  // Auto-connect to WebSocket on app start
  useWebSocketStore.getState().connect(['MISO.HUB.INDIANA', 'PJM.HUB.WEST']);
}
