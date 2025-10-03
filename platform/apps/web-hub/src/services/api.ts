import axios from 'axios';
import { useAuthStore } from '../stores/authStore';

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = useAuthStore.getState().token;
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout();
    }
    return Promise.reject(error);
  }
);

// WebSocket streaming functionality
export class WebSocketManager {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 1000; // Start with 1 second
  private listeners: Map<string, Set<(data: any) => void>> = new Map();
  private isConnected = false;
  private lastInstruments: string[] = [];

  connect(instrumentIds: string[] = []): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/api/v1/stream';
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this.reconnectInterval = 1000;

          // Subscribe to instruments with current token
          this.lastInstruments = instrumentIds;
          this.subscribeToInstruments(this.lastInstruments);

          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        this.ws.onclose = () => {
          console.log('WebSocket disconnected');
          this.isConnected = false;
          this.attemptReconnect(instrumentIds);
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.isConnected = false;
  }

  private attemptReconnect(instrumentIds: string[]): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);

    setTimeout(() => {
      this.connect(instrumentIds).catch(error => {
        console.error('Reconnection failed:', error);
      });
    }, this.reconnectInterval);

    // Exponential backoff
    this.reconnectInterval = Math.min(this.reconnectInterval * 2, 30000);
  }

  private subscribeToInstruments(instrumentIds?: string[]): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const token = useAuthStore.getState().token;
      const subscription = {
        type: 'subscribe',
        instruments: (instrumentIds && instrumentIds.length ? instrumentIds : this.lastInstruments),
        api_key: token || 'dev-key'
      };

      this.ws.send(JSON.stringify(subscription));
    }
  }

  private handleMessage(data: any): void {
    if (data.type === 'price_update' && data.data) {
      // Notify all price update listeners
      this.notifyListeners('price_update', data.data);
    } else if (data.type === 'instrument_update' && data.data) {
      // Notify all instrument update listeners
      this.notifyListeners('instrument_update', data.data);
    }
  }

  private notifyListeners(eventType: string, data: any): void {
    const listeners = this.listeners.get(eventType);
    if (listeners) {
      listeners.forEach(listener => {
        try {
          listener(data);
        } catch (error) {
          console.error('Error in WebSocket listener:', error);
        }
      });
    }
  }

  addListener(eventType: string, callback: (data: any) => void): () => void {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, new Set());
    }

    this.listeners.get(eventType)!.add(callback);

    // Return cleanup function
    return () => {
      const listeners = this.listeners.get(eventType);
      if (listeners) {
        listeners.delete(callback);
        if (listeners.size === 0) {
          this.listeners.delete(eventType);
        }
      }
    };
  }

  isWebSocketConnected(): boolean {
    return this.isConnected;
  }

  // Allow token refresh-triggered re-subscription without reconnecting
  resubscribe(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.subscribeToInstruments();
    }
  }
}

// Global WebSocket manager instance
export const webSocketManager = new WebSocketManager();

