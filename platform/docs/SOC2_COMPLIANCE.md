# SOC2 Compliance Documentation
## 254Carbon Market Intelligence Platform

**Document Version:** 1.0  
**Last Updated:** October 3, 2025  
**Classification:** Confidential

---

## Table of Contents

1. [Overview](#overview)
2. [Security Controls](#security-controls)
3. [Access Control](#access-control)
4. [Data Protection](#data-protection)
5. [Incident Response](#incident-response)
6. [Audit Logging](#audit-logging)
7. [Business Continuity](#business-continuity)
8. [Compliance Matrix](#compliance-matrix)

---

## Overview

This document outlines the security controls and compliance measures implemented in the 254Carbon Market Intelligence Platform to meet SOC2 Type II requirements.

### Scope

- **Services Covered:** All production services in the `market-intelligence` namespace
- **Data Classification:** Confidential market data, customer information, authentication credentials
- **Trust Service Criteria:** Security, Availability, Confidentiality

---

## Security Controls

### CC6.1: Logical and Physical Access Controls

#### Authentication
- **OIDC Integration:** Keycloak-based authentication with OAuth 2.0/OpenID Connect
- **Multi-Factor Authentication:** Required for all admin accounts
- **Session Management:** Token expiration after 8 hours, refresh tokens rotated every 24 hours
- **Password Policy:** Minimum 12 characters, complexity requirements enforced

**Implementation:**
- File: `/platform/apps/gateway/auth.py`
- Configuration: Keycloak realm settings with MFA enabled
- Documentation: User authentication flow diagram

#### Authorization
- **Role-Based Access Control (RBAC):** Implemented at API gateway level
- **Tenant Isolation:** Multi-tenant architecture with strict data separation
- **Entitlements System:** Product/market/channel-level access controls

**Implementation:**
- File: `/platform/apps/gateway/entitlements.py`
- Database: `pg.entitlement_product` table
- Test: CAISO pilot restricted to hub-only data

#### Network Security
- **Network Policies:** Zero-trust architecture with default deny-all
- **IP Allowlisting:** Ingress-level IP restrictions for production API
- **TLS Encryption:** All external traffic encrypted with TLS 1.3
- **Internal mTLS:** Service-to-service communication encrypted

**Implementation:**
- Files: `/platform/infra/k8s/security/network-policies.yaml`
- Ingress: `/platform/infra/k8s/security/ip-allowlist.yaml`
- Certificates: Let's Encrypt with automatic rotation

### CC6.6: Logical Access - Removal and Modification

#### User Lifecycle Management
- **Onboarding:** Approval workflow through Keycloak admin
- **Offboarding:** Immediate access revocation, audit trail maintained
- **Access Reviews:** Quarterly review of user permissions

#### Secrets Management
- **Automated Rotation:** All database passwords and API keys rotated monthly
- **External Secrets Operator:** Integration with AWS Secrets Manager/Vault
- **Encryption at Rest:** All secrets encrypted with AES-256

**Implementation:**
- Files: `/platform/infra/k8s/security/external-secrets.yaml`
- Script: `/platform/infra/scripts/rotate-secrets.sh`
- Schedule: Automated via CronJob (monthly rotation)

### CC7.2: System Monitoring

#### Security Monitoring
- **Intrusion Detection:** ModSecurity WAF at ingress level
- **Anomaly Detection:** Automated detection of brute force, privilege escalation
- **Real-time Alerts:** PagerDuty integration for critical security events

**Implementation:**
- File: `/platform/shared/audit_logger.py` (SecurityEventDetector class)
- Dashboards: `/platform/infra/monitoring/grafana/dashboards/security-audit.json`
- Alerts: Configured for failed auth attempts, authorization denials

#### Performance Monitoring
- **Metrics Collection:** Prometheus scraping all service endpoints
- **Visualization:** Grafana dashboards for SLA tracking
- **Alerting:** Latency, error rate, and uptime alerts

**Implementation:**
- Dashboards: Forecast accuracy, data quality, service health
- SLA Targets: p95 < 250ms, error rate < 1%, uptime > 99.9%

---

## Access Control

### Access Control Matrix

| Role | Data Access | API Access | Admin Functions | Audit Trail |
|------|-------------|------------|-----------------|-------------|
| **System Admin** | Full | Full | Yes | Yes |
| **Data Engineer** | Read/Write | Limited | No | Yes |
| **Analyst User** | Read Only | Full | No | Yes |
| **Pilot Customer (CAISO)** | Hub Only | Downloads | No | Yes |
| **Pilot Customer (MISO)** | Full | Full | No | Yes |

### Principle of Least Privilege

All user accounts are provisioned with minimum necessary permissions. Elevation requires approval and is time-limited.

**Implementation:**
- Kubernetes RBAC policies restrict pod access
- Database users have schema-specific permissions
- Service accounts use non-root UIDs

---

## Data Protection

### Data Classification

| Level | Description | Examples | Controls |
|-------|-------------|----------|----------|
| **Public** | Non-sensitive | API documentation | None |
| **Internal** | Business use | Aggregated statistics | Authentication |
| **Confidential** | Customer data | Market prices, forecasts | Encryption + Audit |
| **Restricted** | Credentials | Database passwords, API keys | Secrets Manager |

### Encryption

#### In Transit
- **External:** TLS 1.3 with strong cipher suites
- **Internal:** mTLS between services (optional, recommended for production)
- **Database:** TLS connections to PostgreSQL and ClickHouse

#### At Rest
- **Databases:** Encryption enabled for PostgreSQL and ClickHouse volumes
- **Object Storage:** Server-side encryption for MinIO/S3
- **Backups:** Encrypted backups with separate key management

### Data Retention

- **Market Data:** 7 years (regulatory requirement)
- **Audit Logs:** 2 years minimum
- **User Data:** Per customer contract
- **Backups:** 30-day retention with daily snapshots

**Implementation:**
- ClickHouse TTL policies for automatic purging
- PostgreSQL pg_cron for data archival
- S3 lifecycle policies for backup retention

---

## Incident Response

### Incident Response Plan

#### Phase 1: Detection (0-15 minutes)
1. Automated alerting triggers incident
2. On-call engineer notified via PagerDuty
3. Initial assessment and severity classification

#### Phase 2: Containment (15-60 minutes)
1. Isolate affected systems
2. Block malicious traffic (IP allowlist updates)
3. Rotate compromised credentials
4. Document all actions in incident log

#### Phase 3: Investigation (1-4 hours)
1. Review audit logs for root cause
2. Identify scope of compromise
3. Determine data exposure
4. Execute recovery procedures

#### Phase 4: Recovery (4-24 hours)
1. Restore from clean backups if needed
2. Apply security patches
3. Verify system integrity
4. Resume normal operations

#### Phase 5: Post-Incident (24-72 hours)
1. Conduct post-mortem analysis
2. Document lessons learned
3. Update security controls
4. Notify affected customers (if required)

### Security Event Classifications

| Severity | Response Time | Example Events |
|----------|---------------|----------------|
| **Critical** | 15 minutes | Data breach, system compromise |
| **High** | 1 hour | Brute force attack, privilege escalation |
| **Medium** | 4 hours | Unusual data export, configuration change |
| **Low** | 24 hours | Failed login attempts, minor policy violations |

### Contact Information

- **Security Team:** security@254carbon.ai
- **On-Call Rotation:** PagerDuty escalation policy
- **Executive Notification:** CTO, CEO (for Critical/High incidents)

---

## Audit Logging

### Comprehensive Audit Trail

All user actions and system events are logged for compliance and investigation purposes.

#### Events Logged

1. **Authentication Events**
   - Login attempts (successful/failed)
   - MFA validation
   - Token refresh/expiration
   - Logout

2. **Authorization Events**
   - Access granted/denied
   - Permission changes
   - Role assignments

3. **Data Access**
   - API calls with parameters
   - Database queries (sensitive data)
   - Data exports
   - Report generation

4. **System Changes**
   - Configuration modifications
   - Service deployments
   - Infrastructure changes
   - Secrets rotation

#### Audit Log Retention

- **Database:** 2 years in PostgreSQL
- **Archive:** 7 years in S3 Glacier
- **Format:** Structured JSON with request/trace IDs
- **Access:** Restricted to Security and Compliance teams

**Implementation:**
- Library: `/platform/shared/audit_logger.py`
- Database Table: `pg.audit_log`
- Dashboard: `/platform/infra/monitoring/grafana/dashboards/security-audit.json`

### Audit Log Integrity

- **Immutability:** Write-only table with no update/delete permissions
- **Tamper Detection:** Checksums calculated for log entries
- **Replication:** Real-time replication to separate audit database
- **Review:** Automated weekly reviews for anomalies

---

## Business Continuity

### Disaster Recovery Plan

#### Recovery Objectives

- **Recovery Time Objective (RTO):** 4 hours
- **Recovery Point Objective (RPO):** 15 minutes
- **Availability Target:** 99.9% uptime

#### Backup Strategy

1. **Database Backups**
   - PostgreSQL: Continuous WAL archiving + daily snapshots
   - ClickHouse: Daily full backups + incremental
   - Retention: 30 days

2. **Configuration Backups**
   - GitOps repository for all Kubernetes manifests
   - Secrets backed up in AWS Secrets Manager
   - Infrastructure as Code in version control

3. **Data Backups**
   - Market data replicated to secondary region
   - S3 cross-region replication for exports
   - Regular restore testing (monthly)

#### Failover Procedures

1. **Database Failover**
   - PostgreSQL: Automatic failover with replication
   - ClickHouse: Manual failover with data sync
   - Time to failover: < 5 minutes

2. **Service Failover**
   - Multi-zone deployment for high availability
   - Health checks and automatic pod restart
   - Load balancer with health probes

3. **Regional Failover**
   - Standby environment in secondary region
   - DNS failover with reduced TTL
   - Time to failover: < 2 hours (manual trigger)

### Testing Schedule

- **Backup Restore Tests:** Monthly
- **Failover Drills:** Quarterly
- **Full DR Exercise:** Annually
- **Tabletop Exercises:** Bi-annually

---

## Compliance Matrix

### SOC2 Trust Service Criteria Mapping

| Criterion | Control | Implementation | Status |
|-----------|---------|----------------|--------|
| **CC6.1** | Logical access controls | OIDC auth, RBAC, MFA | ✅ Implemented |
| **CC6.2** | Authentication | Keycloak with MFA | ✅ Implemented |
| **CC6.3** | Authorization | Entitlements system | ✅ Implemented |
| **CC6.6** | Access removal | Automated offboarding | ✅ Implemented |
| **CC6.7** | Secrets management | External Secrets Operator | ✅ Implemented |
| **CC7.2** | Security monitoring | Grafana dashboards + alerts | ✅ Implemented |
| **CC7.3** | Audit logging | Comprehensive audit trail | ✅ Implemented |
| **CC8.1** | Change management | GitOps + CI/CD pipeline | ✅ Implemented |
| **A1.2** | High availability | Multi-zone deployment | ✅ Implemented |
| **A1.3** | Backup and recovery | Daily backups + DR plan | ✅ Implemented |
| **C1.1** | Data confidentiality | Encryption + access controls | ✅ Implemented |
| **C1.2** | Data disposal | Automated retention policies | ✅ Implemented |

### Compliance Attestation

This platform has been designed and implemented with SOC2 Type II compliance requirements in mind. All listed controls are operational and ready for audit.

**Prepared by:** Platform Engineering Team  
**Reviewed by:** Security & Compliance Team  
**Approved by:** CTO

---

## Appendix

### A. Related Documents

- [Security Policies](./SECURITY_POLICIES.md)
- [Incident Response Playbook](./INCIDENT_RESPONSE.md)
- [Access Control Procedures](./ACCESS_CONTROL.md)
- [Backup and Recovery Guide](./BACKUP_RECOVERY.md)

### B. Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-10-03 | 1.0 | Initial version | Platform Team |

### C. Review Schedule

This document must be reviewed and updated:
- Quarterly for accuracy
- After any security incident
- Before SOC2 audit
- When controls change

---

**End of Document**

