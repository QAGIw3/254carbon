import { create } from 'zustand';
import { webSocketManager } from '../services/api';

type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';

interface OutboundMessage {
  type: string;
  payload: any;
}

export interface EnhancedWebSocketState {
  state: ConnectionState;
  lastError: string | null;
  subscribedInstruments: string[];
  messageQueue: OutboundMessage[];
  metrics: {
    reconnectAttempts: number;
    lastConnectedAt?: number;
    lastDisconnectedAt?: number;
    messagesSent: number;
    messagesQueued: number;
  };

  connect: (instrumentIds?: string[]) => Promise<void>;
  disconnect: () => void;
  subscribe: (instrumentIds: string[]) => void;
  send: (type: string, payload: any) => void;
  flushQueue: () => void;
  setState: (next: ConnectionState) => void;
}

export const useEnhancedWebSocketStore = create<EnhancedWebSocketState>((set, get) => ({
  state: 'disconnected',
  lastError: null,
  subscribedInstruments: [],
  messageQueue: [],
  metrics: {
    reconnectAttempts: 0,
    messagesSent: 0,
    messagesQueued: 0,
  },

  connect: async (instrumentIds: string[] = []) => {
    const current = get().state;
    if (current === 'connecting' || current === 'connected') return;

    set({ state: 'connecting', lastError: null, subscribedInstruments: instrumentIds });
    try {
      await webSocketManager.connect(instrumentIds);
      set(state => ({
        state: 'connected',
        metrics: { ...state.metrics, lastConnectedAt: Date.now(), reconnectAttempts: 0 }
      }));
      get().flushQueue();
    } catch (err: any) {
      set({ state: 'error', lastError: err?.message || 'Connection failed' });
    }
  },

  disconnect: () => {
    webSocketManager.disconnect();
    set(state => ({ state: 'disconnected', metrics: { ...state.metrics, lastDisconnectedAt: Date.now() } }));
  },

  subscribe: (instrumentIds: string[]) => {
    // Reconnect with new subscriptions
    set({ subscribedInstruments: instrumentIds });
    get().disconnect();
    setTimeout(() => get().connect(instrumentIds), 100);
  },

  send: (type: string, payload: any) => {
    // The WebSocketManager currently handles subscribe messages only; we queue others
    const isOpen = (webSocketManager as any).isWebSocketConnected?.() || false;
    if (isOpen) {
      try {
        (webSocketManager as any).ws?.send?.(JSON.stringify({ type, ...payload }));
        set(state => ({ metrics: { ...state.metrics, messagesSent: state.metrics.messagesSent + 1 } }));
      } catch (e) {
        // Fallback to queue on failure
        set(state => ({
          messageQueue: [...state.messageQueue, { type, payload }],
          metrics: { ...state.metrics, messagesQueued: state.metrics.messagesQueued + 1 }
        }));
      }
    } else {
      set(state => ({
        messageQueue: [...state.messageQueue, { type, payload }],
        metrics: { ...state.metrics, messagesQueued: state.metrics.messagesQueued + 1 }
      }));
    }
  },

  flushQueue: () => {
    const queue = get().messageQueue;
    if (!queue.length) return;
    const isOpen = (webSocketManager as any).isWebSocketConnected?.() || false;
    if (!isOpen) return;
    queue.forEach(msg => {
      try {
        (webSocketManager as any).ws?.send?.(JSON.stringify({ type: msg.type, ...msg.payload }));
        set(state => ({ metrics: { ...state.metrics, messagesSent: state.metrics.messagesSent + 1 } }));
      } catch {}
    });
    set({ messageQueue: [] });
  },

  setState: (next: ConnectionState) => {
    if (next === 'reconnecting') {
      set(state => ({ state: next, metrics: { ...state.metrics, reconnectAttempts: state.metrics.reconnectAttempts + 1 } }));
    } else {
      set({ state: next });
    }
  }
}));


