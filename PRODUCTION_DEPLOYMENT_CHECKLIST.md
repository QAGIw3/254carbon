# Production Deployment Checklist

## Pre-Deployment (T-7 Days)

### Infrastructure
- [ ] Kubernetes cluster provisioned and hardened
- [ ] SSL/TLS certificates obtained and configured
- [ ] DNS records configured
- [ ] CDN configured (if applicable)
- [ ] Persistent volumes created with appropriate storage class
- [ ] Backup strategy validated

### Security
- [ ] All secrets rotated
- [ ] Pod security policies applied
- [ ] Network policies configured and tested
- [ ] IP allowlists configured
- [ ] Keycloak realm configured with production settings
- [ ] MFA enabled for all admin accounts
- [ ] Security scan passed (no critical vulnerabilities)

### Data
- [ ] PostgreSQL schemas initialized
- [ ] ClickHouse schemas initialized
- [ ] Historical data backfilled
- [ ] CAISO entitlements configured (API disabled)
- [ ] MISO entitlements configured (full access)
- [ ] Data quality validation passed

### Monitoring
- [ ] Prometheus deployed and scraping all services
- [ ] Grafana dashboards imported
- [ ] Alert rules configured
- [ ] PagerDuty integration tested
- [ ] Log aggregation configured (ELK/Splunk)

## Pre-Deployment (T-48 Hours)

### Application
- [ ] All services built and pushed to registry
- [ ] Image tags locked (no latest tags)
- [ ] Database migrations tested in staging
- [ ] Configuration reviewed and approved
- [ ] Rollback plan documented and tested

### Testing
- [ ] UAT sign-off received from pilot users
- [ ] Load testing completed
- [ ] Disaster recovery tested
- [ ] Failover tested
- [ ] Backup/restore tested

### Documentation
- [ ] User documentation complete
- [ ] API documentation published
- [ ] Runbooks complete
- [ ] On-call rotation established
- [ ] Escalation procedures documented

## Deployment Day

### T-2 Hours: Pre-Flight
- [ ] Code freeze confirmed
- [ ] Full database backup completed
- [ ] Team on standby (Slack/Zoom)
- [ ] Rollback plan reviewed
- [ ] Change request approved

### T-0: Deployment
```bash
# 1. Database migrations
kubectl exec -n market-intelligence postgresql-0 -- \
  psql -U postgres -d market_intelligence -f /migrations/production.sql

# 2. Deploy infrastructure (if needed)
helm upgrade --install infrastructure platform/infra/helm/infrastructure \
  --namespace market-intelligence-infra \
  --values platform/infra/helm/infrastructure/values-prod.yaml

# 3. Deploy application services
helm upgrade --install market-intelligence platform/infra/helm/market-intelligence \
  --namespace market-intelligence \
  --values platform/infra/helm/market-intelligence/values-prod.yaml \
  --set image.tag=$RELEASE_TAG

# 4. Verify deployment
kubectl get pods -n market-intelligence
kubectl get svc -n market-intelligence
```

### T+15min: Smoke Tests
- [ ] Health endpoints responding
- [ ] Authentication working (Keycloak)
- [ ] API Gateway accessible
- [ ] Database connections established
- [ ] Kafka producing/consuming
- [ ] MinIO accessible

### T+30min: Functional Tests
- [ ] MISO API access works
- [ ] CAISO API access blocked (403)
- [ ] Web Hub loads
- [ ] Real-time streaming works (MISO)
- [ ] Forward curves retrievable
- [ ] Data export works
- [ ] Scenario execution works

### T+1hr: Performance Validation
- [ ] API latency <250ms (p95)
- [ ] Stream latency <5s (p95)
- [ ] No error spikes in logs
- [ ] Memory/CPU within normal ranges
- [ ] Database query performance acceptable

### T+2hr: Gradual Rollout
- [ ] 10% traffic to new version
- [ ] Monitor for 30 minutes
- [ ] 50% traffic to new version
- [ ] Monitor for 30 minutes
- [ ] 100% traffic to new version
- [ ] Old version scaled to 0

## Post-Deployment (T+24 Hours)

### Monitoring
- [ ] No alerts triggered
- [ ] SLAs being met (99.9% uptime)
- [ ] Data quality checks passing
- [ ] User feedback positive

### Validation
- [ ] Pilot users confirmed operational
- [ ] All scheduled jobs running (Airflow)
- [ ] Backups completing successfully
- [ ] Audit logs capturing events

### Communication
- [ ] Status page updated
- [ ] Users notified of go-live
- [ ] Support team briefed
- [ ] Success metrics captured

## Rollback Procedure (If Needed)

### Immediate Rollback
```bash
# Rollback application
helm rollback market-intelligence --namespace market-intelligence

# Verify rollback
kubectl get pods -n market-intelligence -w
```

### Database Rollback (If Needed)
```bash
# Restore from backup
pg_restore -U postgres -d market_intelligence /backups/pre-deployment.sql

# Verify data integrity
psql -U postgres -d market_intelligence -c "SELECT COUNT(*) FROM pg.instrument;"
```

### Post-Rollback
- [ ] Root cause analysis initiated
- [ ] Incident report created
- [ ] Fix identified and tested
- [ ] New deployment scheduled

## Success Criteria

- [ ] All pilot users can access the platform
- [ ] CAISO users cannot access API (403 error)
- [ ] MISO users can access API successfully
- [ ] No P0/P1 incidents in first 24 hours
- [ ] SLA targets met (99.9% uptime, <5s latency)
- [ ] Data completeness >99.5%

## Sign-Off

**Engineering Lead**: _________________________ Date: _________

**Product Owner**: _________________________ Date: _________

**Security Officer**: _________________________ Date: _________

**Operations Manager**: _________________________ Date: _________

## Notes

Deployment completed on: __________________

Issues encountered: __________________

Resolution: __________________

Follow-up actions: __________________

