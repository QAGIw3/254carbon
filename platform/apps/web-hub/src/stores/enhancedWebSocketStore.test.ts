import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useEnhancedWebSocketStore } from './enhancedWebSocketStore';

// Mock webSocketManager internals used by the store
vi.mock('../services/api', () => {
  return {
    webSocketManager: {
      connect: vi.fn(() => Promise.resolve()),
      disconnect: vi.fn(() => void 0),
      isWebSocketConnected: vi.fn(() => true),
      ws: { send: vi.fn(() => void 0) },
    },
  };
});

describe('enhancedWebSocketStore', () => {
  beforeEach(() => {
    const { getState, setState } = useEnhancedWebSocketStore;
    setState({
      state: 'disconnected',
      lastError: null,
      subscribedInstruments: [],
      messageQueue: [],
      metrics: { reconnectAttempts: 0, messagesSent: 0, messagesQueued: 0 },
      connect: getState().connect,
      disconnect: getState().disconnect,
      subscribe: getState().subscribe,
      send: getState().send,
      flushQueue: getState().flushQueue,
      setState: getState().setState,
    });
  });

  it('transitions to connected on successful connect', async () => {
    await useEnhancedWebSocketStore.getState().connect(['A']);
    expect(useEnhancedWebSocketStore.getState().state).toBe('connected');
  });

  it('queues messages when not connected and flushes when connected', async () => {
    // start disconnected
    useEnhancedWebSocketStore.getState().send('ping', { ts: Date.now() });
    expect(useEnhancedWebSocketStore.getState().messageQueue.length).toBe(1);

    await useEnhancedWebSocketStore.getState().connect(['A']);
    useEnhancedWebSocketStore.getState().flushQueue();
    expect(useEnhancedWebSocketStore.getState().messageQueue.length).toBe(0);
  });
});


