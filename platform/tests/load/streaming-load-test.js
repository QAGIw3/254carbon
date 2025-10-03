/**
 * K6 WebSocket Streaming Load Test
 * 
 * Tests real-time data streaming against SLA:
 * - Stream latency < 2 seconds
 * - Connection stability
 * - Message throughput
 */

import ws from 'k6/ws';
import { check } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const connectionErrors = new Rate('connection_errors');
const messageLatency = new Trend('message_latency');
const messagesReceived = new Counter('messages_received');
const connectionDuration = new Trend('connection_duration');

export const options = {
  stages: [
    { duration: '1m', target: 50 },   // Ramp to 50 concurrent connections
    { duration: '2m', target: 100 },  // Ramp to 100 connections
    { duration: '5m', target: 100 },  // Hold at 100 connections
    { duration: '2m', target: 200 },  // Spike to 200
    { duration: '2m', target: 200 },  // Hold spike
    { duration: '1m', target: 0 },    // Ramp down
  ],
  
  thresholds: {
    'message_latency': ['p(95)<2000'],  // 95% < 2 seconds
    'connection_errors': ['rate<0.01'], // < 1% connection errors
    'messages_received': ['count>10000'], // Total messages received
  },
};

const WS_URL = __ENV.WS_URL || 'wss://api.254carbon.ai/ws/stream';
const AUTH_TOKEN = __ENV.AUTH_TOKEN || 'test-token';

export default function() {
  const subscriptions = [
    { type: 'price_tick', instruments: ['MISO.NODE.0001', 'MISO.NODE.0002'] },
    { type: 'price_tick', instruments: ['CAISO.TH_SP15_GEN-APND'] },
  ];
  
  const subscription = subscriptions[Math.floor(Math.random() * subscriptions.length)];
  const connectionStart = Date.now();
  
  const params = {
    headers: {
      'Authorization': `Bearer ${AUTH_TOKEN}`,
    },
    tags: { type: 'websocket' },
  };
  
  const res = ws.connect(WS_URL, params, function(socket) {
    socket.on('open', function() {
      console.log('WebSocket connection established');
      
      // Subscribe to price ticks
      socket.send(JSON.stringify({
        action: 'subscribe',
        type: subscription.type,
        instruments: subscription.instruments,
      }));
    });
    
    socket.on('message', function(data) {
      try {
        const message = JSON.parse(data);
        messagesReceived.add(1);
        
        // Calculate latency (message timestamp vs current time)
        if (message.event_time) {
          const eventTime = new Date(message.event_time).getTime();
          const latency = Date.now() - eventTime;
          messageLatency.add(latency);
          
          check(message, {
            'message has valid structure': (m) => m.instrument_id && m.value,
            'latency < 2s': () => latency < 2000,
          });
        }
      } catch (e) {
        console.error('Failed to parse message:', e);
      }
    });
    
    socket.on('error', function(e) {
      console.error('WebSocket error:', e);
      connectionErrors.add(1);
    });
    
    socket.on('close', function() {
      const duration = Date.now() - connectionStart;
      connectionDuration.add(duration);
      console.log('WebSocket connection closed after', duration, 'ms');
    });
    
    // Keep connection alive for 30 seconds
    socket.setTimeout(function() {
      console.log('Closing WebSocket connection after timeout');
      socket.close();
    }, 30000);
  });
  
  check(res, {
    'WebSocket connection successful': (r) => r && r.status === 101,
  });
}


