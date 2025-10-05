# Analytics Hub Consolidation - Deployment Guide

## Quick Start

### 1. Build Docker Images

```bash
# Analytics Service
cd platform/apps/analytics-service
docker build -t 254carbon/analytics-service:latest .

# AI Service
cd platform/apps/ai-service
docker build -t 254carbon/ai-service:latest .
```

### 2. Deploy to Kubernetes

```bash
# Deploy Analytics Service
kubectl apply -f platform/apps/analytics-service/k8s/deployment.yaml

# Deploy AI Service
kubectl apply -f platform/apps/ai-service/k8s/deployment.yaml
```

### 3. Verify Deployments

```bash
# Check pod status
kubectl get pods -n market-intelligence | grep -E "(analytics-service|ai-service)"

# Check service status
kubectl get svc -n market-intelligence | grep -E "(analytics-service|ai-service)"
```

### 4. Test Health Endpoints

```bash
# Analytics Service
kubectl port-forward -n market-intelligence svc/analytics-service 8008:8008
curl http://localhost:8008/health

# AI Service
kubectl port-forward -n market-intelligence svc/ai-service 8020:8020
curl http://localhost:8020/health
```

## Detailed Validation

### Analytics Service Validation

```bash
# Test forecasting endpoint
curl -X POST http://analytics-service:8008/api/v1/ml/forecast \
  -H "Content-Type: application/json" \
  -d '{"instrument_id": "PJM_WEST_HUB", "horizon_months": 12}'

# Test risk analytics
curl -X POST http://analytics-service:8008/api/v1/risk/var \
  -H "Content-Type: application/json" \
  -d '{
    "positions": [{"instrument_id": "PJM_WEST", "quantity": 100}],
    "confidence_level": 0.95,
    "method": "historical"
  }'

# Test market insights
curl "http://analytics-service:8008/api/v1/insights/anomalies?market=PJM&lookback_hours=24"

# Test satellite intelligence
curl "http://analytics-service:8008/api/v1/satellite/providers"

# Test quantum optimization
curl "http://analytics-service:8008/api/v1/quantum/backends"
```

### AI Service Validation

```bash
# Test AI Copilot
curl -X POST http://ai-service:8020/api/v1/copilot/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What drove prices up in PJM yesterday?",
    "language": "en",
    "model": "openai-gpt4"
  }'

# Test NLP service
curl -X POST http://ai-service:8020/api/v1/nlp/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me average prices in CAISO last week"}'

# Test RegTech
curl "http://ai-service:8020/api/v1/regtech/regulations?jurisdiction=ferc_us&since_date=2024-01-01"
```

### WebSocket Test (AI Copilot)

```javascript
// Using wscat or similar tool
wscat -c ws://ai-service:8020/api/v1/copilot/ws/test-conversation

// Send message
> What is the current status of PJM market?

// Receive response
< {"conversation_id": "test-conversation", "response": "..."}
```

## Monitoring

### View Logs

```bash
# Analytics Service logs
kubectl logs -f -n market-intelligence -l app=analytics-service

# AI Service logs
kubectl logs -f -n market-intelligence -l app=ai-service
```

### Resource Usage

```bash
# Check resource utilization
kubectl top pods -n market-intelligence | grep -E "(analytics-service|ai-service)"
```

### Metrics

```bash
# Check Prometheus metrics (if available)
curl http://analytics-service:8008/metrics
curl http://ai-service:8020/metrics
```

## Troubleshooting

### Service Not Starting

1. Check pod events:
```bash
kubectl describe pod -n market-intelligence <pod-name>
```

2. Check logs for errors:
```bash
kubectl logs -n market-intelligence <pod-name>
```

3. Verify environment variables:
```bash
kubectl get pod -n market-intelligence <pod-name> -o yaml | grep -A 10 env:
```

### Database Connection Issues

```bash
# Test ClickHouse connectivity from analytics-service pod
kubectl exec -it -n market-intelligence <analytics-pod> -- \
  python -c "from clickhouse_driver import Client; Client('clickhouse').execute('SELECT 1')"

# Test PostgreSQL connectivity
kubectl exec -it -n market-intelligence <pod-name> -- \
  psql -h postgresql -U postgres -c "SELECT 1"
```

### MLflow Connection Issues

```bash
# Test MLflow connectivity
kubectl exec -it -n market-intelligence <analytics-pod> -- \
  curl http://mlflow:5000/api/2.0/mlflow/experiments/list
```

### LLM API Issues

```bash
# Check if API keys are set
kubectl exec -it -n market-intelligence <ai-pod> -- \
  env | grep -E "(OPENAI|ANTHROPIC)"
```

## Rollback Procedure

If issues arise, rollback to old services:

```bash
# Redeploy old services
kubectl apply -f platform/apps/ml-service/k8s/deployment.yaml
kubectl apply -f platform/apps/risk-service/k8s/deployment.yaml
kubectl apply -f platform/apps/lmp-decomposition-service/k8s/deployment.yaml
kubectl apply -f platform/apps/market-insights/k8s/deployment.yaml
kubectl apply -f platform/apps/nlp-service/k8s/deployment.yaml
kubectl apply -f platform/apps/ai-copilot/k8s/deployment.yaml

# Scale down new services
kubectl scale deployment -n market-intelligence analytics-service --replicas=0
kubectl scale deployment -n market-intelligence ai-service --replicas=0
```

## Performance Benchmarking

### Load Testing

```bash
# Install k6 or similar load testing tool
# Test analytics service
k6 run -  <<EOF
import http from 'k6/http';
import { check } from 'k6';

export const options = {
  vus: 10,
  duration: '30s',
};

export default function () {
  const res = http.get('http://analytics-service:8008/health');
  check(res, { 'status is 200': (r) => r.status === 200 });
}
EOF
```

### Response Time Analysis

```bash
# Measure response times
for i in {1..100}; do
  curl -w "%{time_total}\n" -o /dev/null -s http://analytics-service:8008/health
done | awk '{ sum += $1; n++ } END { print "Average:", sum/n, "seconds" }'
```

## Post-Deployment Checklist

- [ ] Both services are running and healthy
- [ ] All API endpoints return expected responses
- [ ] Database connections are working
- [ ] MLflow connection is functional (analytics-service)
- [ ] LLM APIs are accessible (ai-service)
- [ ] WebSocket connections are stable (ai-service)
- [ ] Intelligence-gateway routes to new services correctly
- [ ] Resource utilization is within acceptable limits
- [ ] No error logs in recent deployment
- [ ] Monitoring dashboards show green status

## Production Deployment Recommendations

1. **Use Blue-Green Deployment**
   - Deploy new services alongside old ones
   - Gradually shift traffic
   - Keep old services running for 1 week as fallback

2. **Enable Monitoring**
   - Set up Prometheus metrics collection
   - Configure Grafana dashboards
   - Set up alerting for errors/latency

3. **Database Migration**
   - If schema changes are needed, run migrations first
   - Test with read replicas before production

4. **API Gateway Configuration**
   - Update route rules gradually
   - Use weighted routing for canary deployment

5. **Backup Strategy**
   - Take snapshots of model registry
   - Backup conversation histories
   - Export MLflow experiments

## Support

For issues or questions:
- Check logs first: `kubectl logs -f <pod-name>`
- Review this guide's troubleshooting section
- Consult service READMEs in respective directories
- Review ANALYTICS_HUB_CONSOLIDATION.md for architecture details

---

**Last Updated:** October 5, 2025  
**Services:** analytics-service:2.0.0, ai-service:2.0.0

