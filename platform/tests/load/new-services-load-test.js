/**
 * K6 Load Test for New Services
 * 
 * Tests performance of newly implemented services:
 * - LMP Decomposition Service
 * - ML Service (Transformer models)
 * - Trading Signals Service
 * - Marketplace Service
 * 
 * Usage:
 *   k6 run new-services-load-test.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const lmpDecompLatency = new Trend('lmp_decomposition_latency');
const mlForecastLatency = new Trend('ml_forecast_latency');
const signalsLatency = new Trend('signals_generation_latency');
const marketplaceLatency = new Trend('marketplace_latency');

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const LMP_SERVICE_URL = __ENV.LMP_SERVICE_URL || 'http://localhost:8009';
const ML_SERVICE_URL = __ENV.ML_SERVICE_URL || 'http://localhost:8006';
const SIGNALS_SERVICE_URL = __ENV.SIGNALS_SERVICE_URL || 'http://localhost:8016';
const MARKETPLACE_SERVICE_URL = __ENV.MARKETPLACE_SERVICE_URL || 'http://localhost:8015';

// Load test options
export const options = {
    stages: [
        { duration: '2m', target: 50 },   // Ramp up to 50 users
        { duration: '5m', target: 50 },   // Stay at 50 users
        { duration: '2m', target: 100 },  // Ramp up to 100 users
        { duration: '5m', target: 100 },  // Stay at 100 users
        { duration: '2m', target: 0 },    // Ramp down
    ],
    thresholds: {
        'http_req_duration': ['p(95)<500'],  // 95% of requests should be below 500ms
        'http_req_failed': ['rate<0.01'],   // Error rate should be below 1%
        'errors': ['rate<0.01'],
        'lmp_decomposition_latency': ['p(95)<300'],
        'ml_forecast_latency': ['p(95)<1000'],  // ML can be slower
        'signals_generation_latency': ['p(95)<200'],
        'marketplace_latency': ['p(95)<150'],
    },
};

// Sample data
const sampleNodes = ['PJM.HUB.WEST', 'PJM.WESTERN', 'PJM.EASTERN', 'PJM.CENTRAL'];
const sampleInstruments = ['PJM.HUB.WEST', 'MISO.HUB.INDIANA', 'ERCOT.HUB.NORTH', 'CAISO.HUB.SP15'];
const strategies = ['mean_reversion', 'momentum', 'spread_trading', 'volatility'];

/**
 * Main scenario - mix of different service calls
 */
export default function() {
    const scenario = Math.floor(Math.random() * 5);

    switch(scenario) {
        case 0:
            testLMPDecomposition();
            break;
        case 1:
            testMLForecast();
            break;
        case 2:
            testTradingSignals();
            break;
        case 3:
            testMarketplace();
            break;
        case 4:
            testCombinedWorkflow();
            break;
    }

    sleep(1);
}

/**
 * Test LMP Decomposition Service
 */
function testLMPDecomposition() {
    const now = new Date();
    const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);

    const payload = JSON.stringify({
        node_ids: [sampleNodes[Math.floor(Math.random() * sampleNodes.length)]],
        start_time: oneHourAgo.toISOString(),
        end_time: now.toISOString(),
        iso: 'PJM',
    });

    const params = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    const res = http.post(`${LMP_SERVICE_URL}/api/v1/lmp/decompose`, payload, params);

    lmpDecompLatency.add(res.timings.duration);

    check(res, {
        'LMP decomposition status 200': (r) => r.status === 200,
        'LMP has components': (r) => {
            try {
                const data = JSON.parse(r.body);
                return data.length > 0 && data[0].energy_component !== undefined;
            } catch {
                return false;
            }
        },
    }) || errorRate.add(1);
}

/**
 * Test ML Forecast Service
 */
function testMLForecast() {
    const payload = JSON.stringify({
        instrument_id: sampleInstruments[Math.floor(Math.random() * sampleInstruments.length)],
        horizon_months: 6,
        features: null,
    });

    const params = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    const res = http.post(`${ML_SERVICE_URL}/api/v1/ml/forecast`, payload, params);

    mlForecastLatency.add(res.timings.duration);

    check(res, {
        'ML forecast status 200 or 404': (r) => r.status === 200 || r.status === 404,  // 404 if no model trained
        'ML forecast has data': (r) => {
            if (r.status === 200) {
                try {
                    const data = JSON.parse(r.body);
                    return data.forecasts && data.forecasts.length > 0;
                } catch {
                    return false;
                }
            }
            return true;  // 404 is acceptable
        },
    }) || errorRate.add(1);
}

/**
 * Test Trading Signals Service
 */
function testTradingSignals() {
    const strategy = strategies[Math.floor(Math.random() * strategies.length)];
    const instrument = sampleInstruments[Math.floor(Math.random() * sampleInstruments.length)];

    // Generate sample price data
    const prices = Array.from({length: 50}, (_, i) => 45 + Math.random() * 10);

    const payload = JSON.stringify({
        strategy: strategy,
        instrument_id: instrument,
        market_data: {
            price: prices[prices.length - 1],
            prices: prices,
        },
    });

    const params = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    const res = http.post(`${SIGNALS_SERVICE_URL}/api/v1/signals/generate`, payload, params);

    signalsLatency.add(res.timings.duration);

    check(res, {
        'Signals status 200': (r) => r.status === 200,
        'Signal has type': (r) => {
            try {
                const data = JSON.parse(r.body);
                return ['BUY', 'SELL', 'HOLD'].includes(data.signal_type);
            } catch {
                return false;
            }
        },
        'Signal has confidence': (r) => {
            try {
                const data = JSON.parse(r.body);
                return data.confidence >= 0 && data.confidence <= 1;
            } catch {
                return false;
            }
        },
    }) || errorRate.add(1);
}

/**
 * Test Marketplace Service
 */
function testMarketplace() {
    // Test different marketplace endpoints
    const endpoint = Math.floor(Math.random() * 3);

    let res;
    switch(endpoint) {
        case 0:
            // List products
            res = http.get(`${MARKETPLACE_SERVICE_URL}/api/v1/marketplace/products`);
            break;
        case 1:
            // Get analytics
            res = http.get(`${MARKETPLACE_SERVICE_URL}/api/v1/marketplace/analytics`);
            break;
        case 2:
            // Create sandbox
            const payload = JSON.stringify({
                partner_id: 'PTR-TEST',
                product_id: 'PRD-TEST-001',
            });
            res = http.post(`${MARKETPLACE_SERVICE_URL}/api/v1/marketplace/sandbox`, payload, {
                headers: { 'Content-Type': 'application/json' },
            });
            break;
    }

    marketplaceLatency.add(res.timings.duration);

    check(res, {
        'Marketplace status 200': (r) => r.status === 200,
        'Marketplace response valid': (r) => {
            try {
                JSON.parse(r.body);
                return true;
            } catch {
                return false;
            }
        },
    }) || errorRate.add(1);
}

/**
 * Test combined workflow across services
 */
function testCombinedWorkflow() {
    const instrument = sampleInstruments[Math.floor(Math.random() * sampleInstruments.length)];

    // 1. Generate trading signal
    const prices = Array.from({length: 50}, (_, i) => 45 + Math.random() * 10);

    const signalPayload = JSON.stringify({
        strategy: 'ml_ensemble',
        instrument_id: instrument,
        market_data: { price: prices[prices.length - 1], prices: prices },
    });

    const signalRes = http.post(
        `${SIGNALS_SERVICE_URL}/api/v1/signals/generate`,
        signalPayload,
        { headers: { 'Content-Type': 'application/json' } }
    );

    check(signalRes, {
        'Signal generated': (r) => r.status === 200,
    }) || errorRate.add(1);

    sleep(0.5);

    // 2. Check marketplace for related analytics products
    const marketRes = http.get(`${MARKETPLACE_SERVICE_URL}/api/v1/marketplace/products?category=analytics`);

    check(marketRes, {
        'Marketplace products fetched': (r) => r.status === 200,
    }) || errorRate.add(1);
}

/**
 * Teardown function
 */
export function handleSummary(data) {
    return {
        'stdout': textSummary(data, { indent: ' ', enableColors: true }),
        'results/new-services-load-test.json': JSON.stringify(data),
    };
}

// Summary helper
function textSummary(data, options) {
    const summary = [];

    summary.push('');
    summary.push('========================================');
    summary.push(' New Services Load Test Summary');
    summary.push('========================================');
    summary.push('');

    const metrics = data.metrics;

    // HTTP metrics
    if (metrics.http_reqs) {
        summary.push(`Total Requests: ${metrics.http_reqs.values.count}`);
    }
    if (metrics.http_req_failed) {
        summary.push(`Failed Requests: ${(metrics.http_req_failed.values.rate * 100).toFixed(2)}%`);
    }

    // Latency metrics
    if (metrics.http_req_duration) {
        summary.push(`Latency p(95): ${metrics.http_req_duration.values['p(95)'].toFixed(2)}ms`);
        summary.push(`Latency p(99): ${metrics.http_req_duration.values['p(99)'].toFixed(2)}ms`);
    }

    // Service-specific metrics
    if (metrics.lmp_decomposition_latency) {
        summary.push('');
        summary.push('LMP Decomposition:');
        summary.push(`  p(95): ${metrics.lmp_decomposition_latency.values['p(95)'].toFixed(2)}ms`);
    }

    if (metrics.ml_forecast_latency) {
        summary.push('');
        summary.push('ML Forecast:');
        summary.push(`  p(95): ${metrics.ml_forecast_latency.values['p(95)'].toFixed(2)}ms`);
    }

    if (metrics.signals_generation_latency) {
        summary.push('');
        summary.push('Trading Signals:');
        summary.push(`  p(95): ${metrics.signals_generation_latency.values['p(95)'].toFixed(2)}ms`);
    }

    if (metrics.marketplace_latency) {
        summary.push('');
        summary.push('Marketplace:');
        summary.push(`  p(95): ${metrics.marketplace_latency.values['p(95)'].toFixed(2)}ms`);
    }

    summary.push('');
    summary.push('========================================');
    summary.push('');

    return summary.join('\n');
}

