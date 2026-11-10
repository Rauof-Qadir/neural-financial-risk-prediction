# Model Risk Management and Deployment Decision

## Decision

**CONDITIONAL_GO_FOR_SHADOW_ONLY**

**Autonomous production decisions: NO_GO**

**Human-reviewed controlled pilot: NOT_YET_APPROVED**

This is a research-project governance assessment, not a formal approval
by a regulated financial institution.

## Evidence reviewed

- Evidence files found: 11 / 11
- Registered risks: 12
- Critical risks: 7
- Readiness dimensions: 9

## Most important known weakness

CASH_OUT fraud between 50K and 250K: 164 of 168 fraud cases were missed in the held-out synthetic test segment.

## Production blockers

- Representative real-world transaction and fraud-label validation.
- Remediation or formally accepted treatment of the weak CASH_OUT segment.
- Live monitoring, incident response and rollback procedures.
- Security, privacy and compliance approval.
- Human review, override and escalation governance.
- End-to-end service, integration and load testing.

## Monitoring requirements

See `monitoring_requirements.json` for proposed monitoring metrics,
frequencies, escalation actions and governance rules.

All thresholds in that specification are provisional. They must be
calibrated and approved using representative institutional data.

## Evidence limitations

- PaySim is synthetic data.
- Gross missed transaction amount is not confirmed financial loss.
- SHAP values explain model behaviour, not causality.
- Drift and stress results are retrospective experiments.
- A shadow deployment still requires privacy, security and operational approval.

## Recommended next step

Build a production-style monitoring and scoring workflow with immutable
model/version tracking, input validation, alert logging, service health
metrics and human-review controls. Keep the frozen model and threshold
unchanged during this engineering phase.
