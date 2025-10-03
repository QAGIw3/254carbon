/**
 * K6 Load Testing Script for 254Carbon Market Intelligence API
 * 
 * Tests API endpoints against SLA targets:
 * - p95 latency < 250ms
 * - Error rate < 1%
 * - Throughput > 100 req/s
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const apiLatency = new Trend('api_latency');
const requestCounter = new Counter('requests');

// Test configuration
export const options = {
  stages: [
    // Ramp up
    { duration: '2m', target: 50 },   // Ramp to 50 users over 2 min
    { duration: '5m', target: 100 },  // Ramp to 100 users over 5 min
    { duration: '5m', target: 200 },  // Ramp to 200 users over 5 min
    
    // Sustained load
    { duration: '10m', target: 200 }, // Hold at 200 users for 10 min
    
    // Spike test
    { duration: '2m', target: 500 },  // Spike to 500 users
    { duration: '3m', target: 500 },  // Hold spike
    
    // Ramp down
    { duration: '2m', target: 0 },    // Ramp down to 0
  ],
  
  thresholds: {
    'http_req_duration{endpoint:instruments}': ['p(95)<250'],  // 95% < 250ms
    'http_req_duration{endpoint:prices}': ['p(95)<250'],
    'http_req_duration{endpoint:curves}': ['p(95)<250'],
    'errors': ['rate<0.01'],  // Error rate < 1%
    'http_req_duration': ['p(99)<500'],  // 99% < 500ms
  },
};

// Configuration
const BASE_URL = __ENV.API_URL || 'https://api.254carbon.ai';
const AUTH_TOKEN = __ENV.AUTH_TOKEN || 'test-token';

// Test scenarios
const instruments = [
  'MISO.NODE.0001',
  'MISO.NODE.0002',
  'CAISO.TH_SP15_GEN-APND',
  'CAISO.TH_NP15_GEN-APND',
];

const markets = ['power', 'gas', 'carbon'];
const products = ['lmp', 'da', 'rt'];

// Setup function - runs once per VU
export function setup() {
  console.log('Starting load test...');
  console.log(`Base URL: ${BASE_URL}`);
  console.log(`Test duration: ~31 minutes`);
  
  // Verify API is accessible
  const res = http.get(`${BASE_URL}/health`);
  check(res, {
    'API is accessible': (r) => r.status === 200,
  });
  
  return { startTime: Date.now() };
}

// Main test function
export default function(data) {
  const headers = {
    'Authorization': `Bearer ${AUTH_TOKEN}`,
    'Content-Type': 'application/json',
  };
  
  // Test 1: Get instruments list
  testInstrumentsList(headers);
  sleep(1);
  
  // Test 2: Get price ticks
  testPriceTicks(headers);
  sleep(1);
  
  // Test 3: Get forward curves
  testForwardCurves(headers);
  sleep(1);
  
  // Test 4: Get fundamentals
  testFundamentals(headers);
  sleep(2);
}

function testInstrumentsList(headers) {
  const market = markets[Math.floor(Math.random() * markets.length)];
  const product = products[Math.floor(Math.random() * products.length)];
  
  const url = `${BASE_URL}/api/v1/instruments?market=${market}&product=${product}`;
  const startTime = Date.now();
  
  const res = http.get(url, { headers, tags: { endpoint: 'instruments' } });
  
  const duration = Date.now() - startTime;
  apiLatency.add(duration);
  requestCounter.add(1);
  
  const success = check(res, {
    'instruments status 200': (r) => r.status === 200,
    'instruments has data': (r) => JSON.parse(r.body).length > 0,
    'instruments latency OK': () => duration < 250,
  });
  
  errorRate.add(!success);
}

function testPriceTicks(headers) {
  const instrument = instruments[Math.floor(Math.random() * instruments.length)];
  
  const url = `${BASE_URL}/api/v1/prices/ticks?` +
    `instrument_id=${instrument}&` +
    `start_time=2025-10-01T00:00:00Z&` +
    `end_time=2025-10-03T00:00:00Z&` +
    `limit=100`;
  
  const startTime = Date.now();
  const res = http.get(url, { headers, tags: { endpoint: 'prices' } });
  const duration = Date.now() - startTime;
  
  apiLatency.add(duration);
  requestCounter.add(1);
  
  const success = check(res, {
    'prices status 200': (r) => r.status === 200,
    'prices has data': (r) => JSON.parse(r.body).data.length > 0,
    'prices latency OK': () => duration < 250,
  });
  
  errorRate.add(!success);
}

function testForwardCurves(headers) {
  const instrument = instruments[Math.floor(Math.random() * instruments.length)];
  
  const url = `${BASE_URL}/api/v1/curves/forward?` +
    `instrument_id=${instrument}&` +
    `as_of_date=2025-10-01&` +
    `scenario_id=BASE`;
  
  const startTime = Date.now();
  const res = http.get(url, { headers, tags: { endpoint: 'curves' } });
  const duration = Date.now() - startTime;
  
  apiLatency.add(duration);
  requestCounter.add(1);
  
  const success = check(res, {
    'curves status 200': (r) => r.status === 200,
    'curves has data': (r) => JSON.parse(r.body).curve_points.length > 0,
    'curves latency OK': () => duration < 250,
  });
  
  errorRate.add(!success);
}

function testFundamentals(headers) {
  const market = 'power';
  
  const url = `${BASE_URL}/api/v1/fundamentals?` +
    `market=${market}&` +
    `start_date=2025-09-01&` +
    `end_date=2025-10-01`;
  
  const startTime = Date.now();
  const res = http.get(url, { headers, tags: { endpoint: 'fundamentals' } });
  const duration = Date.now() - startTime;
  
  apiLatency.add(duration);
  requestCounter.add(1);
  
  const success = check(res, {
    'fundamentals status 200': (r) => r.status === 200,
    'fundamentals latency OK': () => duration < 250,
  });
  
  errorRate.add(!success);
}

// Teardown function
export function teardown(data) {
  const duration = (Date.now() - data.startTime) / 1000;
  console.log(`Load test completed in ${duration.toFixed(2)} seconds`);
}

// Handle test summary
export function handleSummary(data) {
  return {
    'load-test-results.json': JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data, options) {
  const indent = options.indent || '';
  const colors = options.enableColors;
  
  let summary = '\n' + indent + '═══════════════════════════════════════\n';
  summary += indent + '  254Carbon API Load Test Results\n';
  summary += indent + '═══════════════════════════════════════\n\n';
  
  // Request stats
  summary += indent + 'Request Statistics:\n';
  summary += indent + `  Total Requests: ${data.metrics.requests.values.count}\n`;
  summary += indent + `  Failed Requests: ${data.metrics.http_req_failed.values.passes}\n`;
  summary += indent + `  Error Rate: ${(data.metrics.errors.values.rate * 100).toFixed(2)}%\n\n`;
  
  // Latency stats
  summary += indent + 'Latency Statistics:\n';
  summary += indent + `  Average: ${data.metrics.http_req_duration.values.avg.toFixed(2)}ms\n`;
  summary += indent + `  Median (p50): ${data.metrics.http_req_duration.values['p(50)'].toFixed(2)}ms\n`;
  summary += indent + `  p95: ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms\n`;
  summary += indent + `  p99: ${data.metrics.http_req_duration.values['p(99)'].toFixed(2)}ms\n`;
  summary += indent + `  Max: ${data.metrics.http_req_duration.values.max.toFixed(2)}ms\n\n`;
  
  // SLA validation
  summary += indent + 'SLA Validation:\n';
  const p95 = data.metrics.http_req_duration.values['p(95)'];
  const errorRate = data.metrics.errors.values.rate;
  
  summary += indent + `  ✓ p95 < 250ms: ${p95 < 250 ? 'PASS' : 'FAIL'} (${p95.toFixed(2)}ms)\n`;
  summary += indent + `  ✓ Error rate < 1%: ${errorRate < 0.01 ? 'PASS' : 'FAIL'} (${(errorRate * 100).toFixed(2)}%)\n\n`;
  
  summary += indent + '═══════════════════════════════════════\n';
  
  return summary;
}


