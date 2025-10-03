# User Acceptance Testing (UAT) Plan

## Overview

UAT for the 254Carbon Market Intelligence Platform with MISO and CAISO pilot customers.

**Duration**: 4 weeks  
**Participants**: 2 pilot organizations (5 MISO users, 3 CAISO users)  
**Environment**: Dedicated UAT environment (staging)

## Objectives

1. Validate functionality meets user requirements
2. Confirm entitlement restrictions (CAISO: Hub+Downloads only)
3. Test real-world workflows
4. Identify usability issues
5. Verify performance meets SLAs
6. Gather feedback for improvements

## Test Environment

### UAT Staging

```bash
# Deploy UAT environment
helm upgrade --install market-intelligence-uat platform/infra/helm/market-intelligence \
  --namespace market-intelligence-uat \
  --set environment=uat \
  --set replicaCount=2
```

**Access**:
- Web Hub: https://uat.254carbon.ai
- API Gateway: https://api-uat.254carbon.ai
- Grafana: https://grafana-uat.254carbon.ai

### Test Data

- **MISO**: 90 days historical LMP data (1000 nodes)
- **CAISO**: 90 days historical settled prices (500 nodes)
- **Forecasts**: Baseline scenario curves to 2050
- **Fundamentals**: Load, generation, capacity data

## Pilot Organizations

### MISO Pilot Customer

**Organization**: MidAmerica Energy Trading LLC  
**Users**: 5 traders + analysts  
**Entitlements**: Full access (Hub + API + Downloads)  
**Focus**: Real-time pricing, forward curves, API integration

### CAISO Pilot Customer

**Organization**: Pacific Power Solutions Inc.  
**Users**: 3 risk analysts  
**Entitlements**: Hub + Downloads only (NO API)  
**Focus**: Daily pricing, scenario analysis, report downloads

## Test Scenarios

### Scenario 1: Authentication and Access

**MISO**:
1. Log in via SSO (Keycloak)
2. Verify Dashboard loads with MISO data
3. Confirm API access (generate token, test `/instruments` endpoint)
4. Test download CSV export

**CAISO**:
1. Log in via SSO
2. Verify Dashboard loads with CAISO data
3. **Confirm API access is BLOCKED** (403 error expected)
4. Test download Parquet export

**Success Criteria**:
- All users can authenticate
- CAISO users receive 403 on API attempts
- MISO users can access API successfully

### Scenario 2: Real-Time Data Streaming

**MISO only** (CAISO has no RT data):
1. Navigate to Market Explorer
2. Subscribe to 10 nodal prices
3. Verify updates arrive within 5 seconds
4. Monitor for 15 minutes
5. Check for data gaps or errors

**Success Criteria**:
- Stream latency p95 ≤ 5s
- No missing intervals
- Clean disconnection on page close

### Scenario 3: Forward Curve Analysis

**Both MISO & CAISO**:
1. Navigate to Curves page
2. Select Hub (MISO: Indiana Hub, CAISO: SP15)
3. View monthly curve to 2050
4. Compare baseline vs alternative scenario
5. Export curve data to CSV
6. Import CSV into Excel for analysis

**Success Criteria**:
- Curves load in <2 seconds
- All tenors visible (monthly, quarterly, annual)
- CSV format is correct and importable

### Scenario 4: Scenario Modeling

**Both MISO & CAISO**:
1. Navigate to Scenario Builder
2. Create new scenario: "High Load Growth"
3. Adjust assumptions:
   - Load growth: +2.5% CAGR
   - Gas price: +$1.50/MMBtu
4. Submit scenario run
5. Wait for completion
6. View updated curves
7. Compare to baseline

**Success Criteria**:
- Scenario creation intuitive
- Run completes in <10 minutes
- Results clearly differentiated from baseline

### Scenario 5: Data Export (Downloads)

**Both MISO & CAISO**:
1. Navigate to Downloads page
2. Select multiple instruments (20 nodes)
3. Choose date range (last 30 days)
4. Select format: Parquet
5. Request export
6. Download file via signed URL
7. Validate file contents

**Success Criteria**:
- Export completes in <5 minutes
- Signed URL valid for 1 hour
- Parquet file readable in Python/Pandas
- Data completeness ≥99.5%

### Scenario 6: API Integration (MISO only)

**MISO Traders**:
1. Obtain API key from Hub
2. Test authentication (Bearer token)
3. Query `/api/v1/instruments?market=power`
4. Query `/api/v1/prices/ticks` for specific nodes
5. Query `/api/v1/curves/forward` for hub curves
6. Implement simple Python script to fetch daily curves
7. Run script via cron job

**Success Criteria**:
- API documentation clear
- All endpoints return expected data
- Rate limits clearly communicated (1000/hr)
- Python script runs reliably

### Scenario 7: Report Generation

**Both MISO & CAISO**:
1. Navigate to Reports (if available) or use Download
2. Generate "Monthly Market Summary" report
3. Review HTML preview
4. Download PDF
5. Verify charts and tables render correctly

**Success Criteria**:
- Report generation <2 minutes
- PDF quality suitable for presentation
- Data accuracy matches source

### Scenario 8: Performance Under Load

**Concurrent Users**:
- 5 MISO users + 3 CAISO users online simultaneously
- All performing various operations
- Monitor system performance

**Success Criteria**:
- No degradation in response times
- API p95 latency <250ms
- Web Hub pages load <2s
- No errors or timeouts

## Acceptance Criteria

### Functional Requirements

| Requirement | Pass/Fail | Notes |
|-------------|-----------|-------|
| User authentication (SSO) | | |
| MISO: API access enabled | | |
| CAISO: API access blocked | | |
| Real-time data streaming (MISO) | | |
| Forward curve visualization | | |
| Scenario creation and execution | | |
| Data exports (CSV, Parquet) | | |
| Signed download URLs | | |
| Report generation | | |

### Non-Functional Requirements

| Requirement | Target | Measured | Pass/Fail |
|-------------|--------|----------|-----------|
| Stream latency (p95) | ≤5s | | |
| API latency (p95) | ≤250ms | | |
| Data freshness (nodal) | ≤5 min | | |
| Data completeness | ≥99.5% | | |
| Uptime | ≥99.9% | | |
| Forecast MAPE (months 1-6) | ≤12% | | |

### Usability

- [ ] Navigation intuitive (average user finds features <3 clicks)
- [ ] Documentation sufficient (users can self-serve)
- [ ] Error messages clear and actionable
- [ ] Onboarding process smooth (<30 min to first value)

## Feedback Collection

### Daily Standup

- **Time**: 10 AM EST daily
- **Participants**: UAT users + 254Carbon team
- **Agenda**:
  1. Blockers from previous day
  2. Today's test focus
  3. Quick feedback

### Feedback Form

Google Form with questions:
1. What task were you attempting?
2. Did it work as expected? (Yes/No/Partially)
3. What was confusing or difficult?
4. What would you improve?
5. Overall satisfaction (1-5 stars)

### Weekly Retrospective

- **Time**: Friday 2 PM EST
- **Format**: Open discussion
- **Topics**:
  - Biggest wins this week
  - Top 3 issues
  - Feature requests
  - General impressions

## Issue Tracking

All issues logged in GitLab:

**Labels**:
- `uat::blocker` - Prevents testing
- `uat::critical` - Major functionality broken
- `uat::enhancement` - Nice-to-have improvement
- `uat::documentation` - Docs issue

**SLA**:
- Blockers: 4 hours
- Critical: 24 hours
- Enhancements: Best effort

## Exit Criteria

UAT is complete when:

1. **All critical scenarios pass** (Scenarios 1-7)
2. **Performance targets met** (SLAs achieved)
3. **No P0/P1 bugs open**
4. **Pilot users sign-off** (formal approval)
5. **Documentation approved** (user guide, API docs)
6. **Training completed** (all users certified)

## Go/No-Go Decision

**Meeting**: End of Week 4  
**Attendees**: Pilot users, Product Owner, Engineering Lead, Security Officer

**Checklist**:
- [ ] All acceptance criteria met
- [ ] No outstanding critical issues
- [ ] Pilot users comfortable with production use
- [ ] Security audit passed
- [ ] DR test successful
- [ ] Runbooks complete
- [ ] On-call rotation established

## Post-UAT Production Deployment

### Deployment Plan

1. **Code Freeze**: 48 hours before deployment
2. **Pre-Deployment**:
   - Full database backup
   - Verify rollback plan
   - Pre-warm caches
3. **Deployment Window**: Saturday 2 AM - 6 AM EST
4. **Deployment Steps**:
   ```bash
   # 1. Database migrations
   kubectl exec -n market-intelligence postgresql-0 -- \
     psql -U postgres -d market_intelligence -f migrations/prod-deploy.sql
   
   # 2. Deploy services (blue-green)
   helm upgrade market-intelligence platform/infra/helm/market-intelligence \
     --namespace market-intelligence \
     --set image.tag=$RELEASE_TAG \
     --set environment=production
   
   # 3. Smoke tests
   ./tests/smoke-tests-prod.sh
   
   # 4. Enable traffic (gradual)
   kubectl patch svc api-gateway -p '{"spec":{"selector":{"version":"new"}}}'
   ```
5. **Post-Deployment**:
   - Monitor for 2 hours
   - Gradual user migration (10% → 50% → 100%)
   - Final sign-off

### Rollback Plan

If critical issues detected:
```bash
# Instant rollback
helm rollback market-intelligence --namespace market-intelligence

# Restore database if needed
pg_restore -U postgres -d market_intelligence backup.sql
```

## Training

### User Training Sessions

**Week 1 of UAT**:
- Session 1: Platform overview (1 hour)
- Session 2: Hands-on navigation (2 hours)
- Session 3: API integration (MISO only, 2 hours)

**Materials**:
- Video recordings
- Quick start guide (PDF)
- API documentation
- FAQ

### Admin Training

For pilot organization admins:
- User management
- Entitlement configuration
- Usage monitoring
- Support escalation

## Success Metrics

### Quantitative

- **Task Completion Rate**: >90%
- **Time to Complete Tasks**: Within expected range
- **Error Rate**: <5%
- **Performance**: All SLAs met
- **User Satisfaction**: Average ≥4/5 stars

### Qualitative

- Users express confidence in production use
- Positive feedback on usability
- Willingness to recommend to peers
- Excitement about future features

## UAT Schedule

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | Setup, training, basic functionality | Users onboarded, credentials issued |
| 2 | Core workflows (Scenarios 1-4) | Feedback report #1 |
| 3 | Advanced features (Scenarios 5-7) | Feedback report #2 |
| 4 | Performance, integration, final testing | Go/No-Go decision |

## Contact

**UAT Coordinator**: Sarah Johnson (sarah@254carbon.ai)  
**Technical Support**: support@254carbon.ai  
**Slack Channel**: #uat-pilot  
**Office Hours**: Mon-Fri 9 AM - 5 PM EST

