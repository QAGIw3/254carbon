# Security Implementation

## Overview

This document describes the security controls and compliance measures implemented in the 254Carbon Market Intelligence Platform.

## Authentication & Authorization

### Keycloak OIDC

- **Provider**: Keycloak 23.0
- **Protocol**: OpenID Connect (OIDC)
- **Token Type**: JWT with RS256 signing
- **Session Management**: Server-side sessions with 8-hour timeout
- **MFA**: Supported (TOTP, WebAuthn)

### Service Accounts

- **Type**: OAuth2 Client Credentials flow
- **Scopes**: Granular permissions (read:ticks, write:scenarios, etc.)
- **Rotation**: 90-day mandatory rotation

### API Security

- **Authentication**: Bearer token (JWT) required for all API endpoints
- **Rate Limiting**: 1000 requests/hour per user, 10000/hour per organization
- **CORS**: Restricted to approved origins
- **HTTPS Only**: TLS 1.3 enforced

## Authorization Model

### Entitlement System

Multi-dimensional access control:

1. **Market** (power, gas, env, lng)
2. **Product** (lmp, curve, rec, etc.)
3. **Channel** (hub, api, downloads)
4. **Instrument** (specific locations/hubs)
5. **Time Window** (from_date, to_date)

### CAISO Pilot Restrictions

```sql
-- CAISO: Hub + Downloads only (API disabled)
UPDATE pg.entitlement_product
SET channels = '{"hub": true, "api": false, "downloads": true}'::jsonb
WHERE market = 'power' 
  AND product = 'lmp'
  AND tenant_id = 'pilot_caiso';
```

### MISO Pilot Access

```sql
-- MISO: Full access (Hub + API + Downloads)
UPDATE pg.entitlement_product
SET channels = '{"hub": true, "api": true, "downloads": true}'::jsonb
WHERE market = 'power' 
  AND product = 'lmp'
  AND tenant_id = 'pilot_miso';
```

## Audit Logging

### Coverage

All operations are logged:

- Authentication attempts (success/failure)
- API calls (endpoint, params, user, timestamp, latency)
- Data exports (instruments, format, size)
- Scenario executions (scenario_id, run_id)
- Admin actions (entitlement changes, user management)

### Audit Log Schema

```sql
CREATE TABLE pg.audit_log (
    audit_id      UUID PRIMARY KEY,
    timestamp     TIMESTAMPTZ,
    user_id       TEXT,
    tenant_id     TEXT,
    action        TEXT,
    resource_type TEXT,
    resource_id   TEXT,
    ip_address    INET,
    user_agent    TEXT,
    success       BOOLEAN,
    details       JSONB
);
```

### Retention

- **Active Logs**: 1 year in PostgreSQL
- **Archive**: 7 years in S3 (compliance)
- **Hot Search**: Last 90 days indexed in Elasticsearch

## Data Encryption

### At Rest

- **PostgreSQL**: AES-256 encryption enabled
- **ClickHouse**: Column-level encryption for sensitive fields
- **MinIO**: Server-side encryption (SSE-S3)
- **Secrets**: Kubernetes secrets with etcd encryption

### In Transit

- **TLS 1.3** for all network communication
- **Certificate Management**: Let's Encrypt + cert-manager
- **Internal Services**: mTLS via service mesh (Istio)

## Secrets Management

### Kubernetes Secrets

- **Storage**: etcd with encryption at rest
- **Access**: RBAC-controlled, namespace-scoped
- **Rotation**: Automated monthly via CronJob

### Rotation Schedule

| Secret Type | Rotation Frequency | Automation |
|-------------|-------------------|------------|
| Database passwords | 30 days | Automated |
| API keys | 90 days | Automated |
| TLS certificates | 90 days (Let's Encrypt) | Automated |
| Service account tokens | 90 days | Automated |
| Keycloak master password | 180 days | Manual |

### Secret Rotation Process

```bash
# Automated via CronJob (monthly)
kubectl apply -f platform/infra/k8s/secrets-rotation-cronjob.yaml
```

## Network Security

### Kubernetes Network Policies

**Default Deny**:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

**Service-Specific Allow**:
- API Gateway can access: ClickHouse, PostgreSQL, Kafka
- Curve Service can access: ClickHouse, PostgreSQL
- Web Hub can access: API Gateway only

### IP Allowlisting

```yaml
# Example: Restrict API access to office IPs
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-gateway-ingress
spec:
  podSelector:
    matchLabels:
      app: api-gateway
  ingress:
  - from:
    - ipBlock:
        cidr: 203.0.113.0/24  # Office IP range
    ports:
    - protocol: TCP
      port: 8000
```

## Pod Security

### Pod Security Standards

- **Enforced**: Restricted profile
- **Non-root users**: All containers run as UID 1000+
- **Read-only root filesystem**: Where possible
- **No privilege escalation**: allowPrivilegeEscalation: false
- **Capabilities dropped**: Drop ALL, add only required

### Example Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: gateway
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
```

## Vulnerability Management

### Container Scanning

**Trivy** integration in CI/CD:
```yaml
scan:security:
  stage: scan
  image: aquasec/trivy:latest
  script:
    - trivy image 254carbon/gateway:$CI_COMMIT_SHA
  allow_failure: false  # Block on HIGH/CRITICAL
```

### Dependency Scanning

- **Python**: Safety, Bandit
- **Node.js**: npm audit
- **Base Images**: Official, minimal images only

### Patch Management

- **Base Images**: Weekly rebuild with latest patches
- **Dependencies**: Dependabot automated PRs
- **CVE Response**: 48-hour SLA for CRITICAL, 7-day for HIGH

## Incident Response

### Detection

- **SIEM**: Splunk integration for audit logs
- **Alerting**: PagerDuty for security events
- **Anomaly Detection**: ML-based (Elastic Security)

### Response Procedures

1. **Detection**: Automated alerts + 24/7 SOC monitoring
2. **Triage**: P0 (15 min), P1 (1 hour), P2 (4 hours)
3. **Containment**: Network isolation, service shutdown
4. **Eradication**: Patch, rotate secrets, rebuild
5. **Recovery**: Gradual rollout with monitoring
6. **Post-Mortem**: Within 72 hours

## Compliance

### SOC 2 Type II

**Control Mappings**:

| Control Domain | Implementation |
|----------------|----------------|
| Access Control | OIDC, RBAC, MFA, entitlements |
| Audit Logging | Comprehensive audit log (1yr retention) |
| Encryption | TLS 1.3, AES-256 at rest |
| Change Management | GitOps, PR reviews, CI/CD |
| Backup & Recovery | Daily backups, RPO 15min, RTO 60min |
| Incident Response | Documented procedures, on-call rotation |

### GDPR Considerations

- **Right to Access**: API endpoint for user data export
- **Right to Erasure**: Soft-delete with anonymization
- **Data Minimization**: Only collect necessary fields
- **Consent Management**: Explicit opt-in for analytics

### Audit Evidence

Automated collection for compliance:
- All code changes (GitLab)
- Access logs (PostgreSQL audit_log)
- Configuration changes (Kubernetes events)
- Security scans (Trivy reports)

## Security Checklist

### Pre-Deployment

- [ ] All secrets rotated
- [ ] TLS certificates valid
- [ ] Network policies applied
- [ ] Pod security policies enforced
- [ ] Vulnerability scans passed
- [ ] Penetration testing completed

### Post-Deployment

- [ ] Audit logging verified
- [ ] Monitoring alerts configured
- [ ] Incident response playbook reviewed
- [ ] Access controls tested
- [ ] Backup/restore validated

## Contact

**Security Team**: security@254carbon.ai  
**Responsible Disclosure**: security@254carbon.ai (PGP key available)  
**Bug Bounty**: HackerOne program (private)

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CIS Kubernetes Benchmark](https://www.cisecurity.org/benchmark/kubernetes)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [SOC 2 Trust Services Criteria](https://www.aicpa.org/interestareas/frc/assuranceadvisoryservices/socforserviceorganizations.html)

