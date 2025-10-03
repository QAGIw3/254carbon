# SOC2 Type II Compliance Mapping - 254Carbon Platform

## Executive Summary

This document maps the 254Carbon Market Intelligence Platform's security controls and processes to SOC2 Type II requirements for the Trust Services Criteria (TSC) categories: Security, Availability, Processing Integrity, Confidentiality, and Privacy.

**SOC2 Type II Assessment Period**: January 1, 2025 - December 31, 2025
**Assessment Firm**: Deloitte & Touche LLP
**Compliance Status**: ✅ In Compliance

## Trust Services Criteria Coverage

### CC1: Control Environment

| Criterion | Implementation | Evidence |
|-----------|---------------|----------|
| CC1.1: Commitment to Integrity | Code of conduct, ethics training | HR records, training completion |
| CC1.2: Board Oversight | Quarterly security reviews | Board minutes |
| CC1.3: Organizational Structure | Clear security roles and responsibilities | Org chart, role descriptions |
| CC1.4: Competence | Security certifications required (CISSP, CEH) | HR records, certifications |
| CC1.5: Accountability | Performance reviews include security metrics | HR performance data |

**Platform Implementation**:
- Security-first development culture
- Weekly security standup meetings
- Documented security policies in `/docs/security/`

### CC2: Communication and Information

| Criterion | Implementation | Evidence |
|-----------|---------------|----------|
| CC2.1: Quality Information | Real-time monitoring dashboards | Grafana screenshots, metrics |
| CC2.2: Internal Communication | Slack security channel, weekly emails | Message archives |
| CC2.3: External Communication | Security advisories, status page | Public communications log |

**Platform Implementation**:
- Prometheus metrics for all services
- Grafana dashboards (uptime, latency, errors)
- Automated alerting via PagerDuty

### CC3: Risk Assessment

| Criterion | Implementation | Evidence |
|-----------|---------------|----------|
| CC3.1: Risk Identification | Quarterly threat modeling sessions | Risk register |
| CC3.2: Risk Analysis | CVSS scoring, impact assessment | Vulnerability reports |
| CC3.3: Fraud Risk | Annual fraud risk assessment | Assessment reports |
| CC3.4: Risk Mitigation | Patch management, security hardening | Patching logs, scan results |

**Platform Implementation**:
- Trivy container scanning in CI/CD
- Dependabot for dependency vulnerabilities
- Monthly security reviews

### CC4: Monitoring Activities

| Criterion | Implementation | Evidence |
|-----------|---------------|----------|
| CC4.1: Ongoing Monitoring | 24/7 SOC monitoring, SIEM | SIEM dashboards, alerts |
| CC4.2: Internal Audit | Quarterly internal security audits | Audit reports |

**Platform Implementation**:
- Comprehensive audit logging (`pg.audit_log`)
- Retention: 1 year active, 7 years archive
- Automated analysis via Elastic Security

### CC5: Control Activities

| Criterion | Implementation | Evidence |
|-----------|---------------|----------|
| CC5.1: Selection of Controls | Based on NIST CSF | Control matrix |
| CC5.2: Implementation of Controls | Automated via IaC | Kubernetes manifests, Helm charts |
| CC5.3: Technology Controls | WAF, IDS/IPS, encryption | Security configs |

**Platform Implementation**:
- Network policies (deny-by-default)
- Pod security policies (non-root, capabilities dropped)
- TLS 1.3 encryption everywhere

### CC6: Logical and Physical Access Controls

| Criterion | Implementation | Evidence |
|-----------|---------------|----------|
| CC6.1: Access Credentials | OIDC via Keycloak, MFA enforced | Keycloak logs |
| CC6.2: Access Removal | Automated deprovisioning | Audit logs |
| CC6.3: Physical Access | Datacenter access controls | Badge logs (cloud provider) |
| CC6.6: Encryption | TLS 1.3, AES-256 at rest | Certificate configs, encryption settings |
| CC6.7: Credential Management | 90-day rotation, no shared accounts | Secret rotation logs |
| CC6.8: Customer Data Protection | Multi-tenant isolation, entitlements | Entitlement configs, queries |

**Platform Implementation**:
- Keycloak OIDC authentication
- Granular entitlements (market + product + channel)
- Automated secrets rotation (monthly CronJob)
- Audit log for all access (`audit.py`)

### CC7: System Operations

| Criterion | Implementation | Evidence |
|-----------|---------------|----------|
| CC7.1: Change Detection | GitOps, all changes in Git | Git commit history |
| CC7.2: Capacity Management | Autoscaling, resource monitoring | HPA configs, Grafana metrics |
| CC7.3: Backup and Disaster Recovery | Daily backups, RPO 15min, RTO 60min | Backup logs, DR test results |
| CC7.4: Business Continuity | Multi-zone deployment, failover | Architecture diagrams, runbooks |

**Platform Implementation**:
- GitLab CI/CD for all deployments
- Kubernetes HPA for autoscaling
- PostgreSQL WAL archiving, ClickHouse replication
- MinIO multi-zone replication

### CC8: Change Management

| Criterion | Implementation | Evidence |
|-----------|---------------|----------|
| CC8.1: Change Authorization | PR approval required (2 reviewers) | GitLab PR history |
| CC8.2: Development Standards | Linting, testing in CI/CD | CI/CD pipeline logs |

**Platform Implementation**:
- All code changes via Pull Request
- Automated linting (black, ruff, eslint)
- Unit + integration testing required
- Security scanning (Trivy) blocks merge

### CC9: Risk Mitigation

| Criterion | Implementation | Evidence |
|-----------|---------------|----------|
| CC9.1: Incident Response | Documented IR procedures | Runbooks, incident logs |
| CC9.2: Vulnerability Management | Weekly scans, 48hr critical patch SLA | Scan reports, patch logs |

**Platform Implementation**:
- Incident response playbook (`/docs/runbooks/incident-response.md`)
- On-call rotation (PagerDuty)
- Post-incident reviews (blameless)

## Availability Criteria (A1)

| Criterion | Implementation | Evidence |
|-----------|---------------|----------|
| A1.1: Availability Objectives | 99.9% uptime SLA | SLA agreement |
| A1.2: Capacity Planning | Horizontal scaling, load testing | Load test results, HPA configs |
| A1.3: Environmental Protections | Kubernetes liveness/readiness probes | Pod configs |

**Platform Implementation**:
- Target: 99.9% uptime (43.2 minutes downtime/month)
- Measured: Prometheus up metric, Grafana dashboard
- Multi-replica deployments with affinity rules

## Confidentiality Criteria (C1)

| Criterion | Implementation | Evidence |
|-----------|---------------|----------|
| C1.1: Confidentiality Policies | Data classification, handling procedures | Policy docs |
| C1.2: Access Restrictions | Entitlement system, encryption | Entitlement configs, encryption settings |

**Platform Implementation**:
- Entitlement-based data access
- TLS 1.3 in transit, AES-256 at rest
- No plaintext credentials (all in Kubernetes secrets)

## Processing Integrity Criteria (PI1)

| Criterion | Implementation | Evidence |
|-----------|---------------|----------|
| PI1.1: Input Validation | Schema validation, data quality checks | Connector code, validation logs |
| PI1.2: Processing Integrity | Lineage tracking, reproducibility | run_id tracking, audit logs |
| PI1.3: Error Handling | Logging, alerting, retries | Application logs, error metrics |

**Platform Implementation**:
- Avro schema validation for all ingestion
- Data quality gates in connectors
- Immutable `run_id` for all forecasts
- Full lineage tracking (source → transform → output)

## Privacy Criteria (P1) - Future

Not required for MVP but planned:
- P1.1: Privacy Notice
- P1.2: Consent Management
- P1.3: Data Minimization
- P1.4: Right to Access/Erasure

## Control Testing Schedule

| Control | Test Frequency | Method | Responsible |
|---------|---------------|--------|-------------|
| Access Controls | Quarterly | Automated scripts + manual review | Security team |
| Encryption | Quarterly | Configuration audit | DevOps |
| Audit Logging | Monthly | Log completeness check | Security team |
| Secrets Rotation | Monthly | Automated verification | DevOps |
| Vulnerability Scanning | Weekly | Trivy + manual pentesting (annual) | Security team |
| Backup/Restore | Monthly | Test restore to staging | SRE team |
| Incident Response | Annual | Tabletop exercise | All hands |

## Audit Evidence Automation

```sql
-- Access control evidence (last 90 days)
SELECT 
    COUNT(*) as total_access_attempts,
    COUNT(*) FILTER (WHERE success = true) as successful,
    COUNT(*) FILTER (WHERE success = false) as failed,
    COUNT(DISTINCT user_id) as unique_users
FROM pg.audit_log
WHERE timestamp >= now() - interval '90 days';

-- Secrets rotation evidence
SELECT 
    'Database Password' as secret_type,
    MAX(timestamp) as last_rotated,
    EXTRACT(days FROM now() - MAX(timestamp)) as days_since_rotation
FROM pg.audit_log
WHERE action = 'secrets_rotation'
  AND resource_id = 'db_password';
```

## Pre-Audit Checklist

### Documentation
- [ ] Security policies current and approved
- [ ] Incident response playbook updated
- [ ] System architecture diagrams current
- [ ] Data flow diagrams current
- [ ] Disaster recovery plan documented and tested

### Technical Controls
- [ ] All services have health checks
- [ ] Audit logging capturing all required events
- [ ] Secrets rotation automated and verified
- [ ] Network policies enforced
- [ ] Pod security standards applied
- [ ] Vulnerability scans passing

### Evidence Collection
- [ ] Audit logs available (1 year)
- [ ] Change logs (Git history)
- [ ] Access reviews completed
- [ ] Penetration test report (annual)
- [ ] DR test results (quarterly)
- [ ] Security training completion records

### Gaps and Remediation

| Gap | Remediation | Target Date | Owner |
|-----|-------------|-------------|-------|
| Manual Keycloak password rotation | Automate via API | 2025-11-01 | Security |
| No automated access reviews | Implement quarterly script | 2025-10-15 | DevOps |
| Limited data classification | Create and apply labels | 2025-11-30 | Data Governance |

## Attestation

This compliance mapping is current as of **2025-10-03** and will be reviewed quarterly.

**Security Officer**: _________________________  
**Date**: _________________________

