
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports" / "model_governance"
OUTPUT.mkdir(parents=True, exist_ok=True)


# Evidence files from completed experiments.
EVIDENCE = {
    "frozen_test": (
        "reports/final_evaluation/final_test_results.json"
    ),
    "model_freeze": (
        "reports/model_freeze/model_freeze_manifest.json"
    ),
    "error_analysis": (
        "reports/error_analysis/error_summary.json"
    ),
    "missed_exposure": (
        "reports/missed_fraud_exposure/"
        "missed_fraud_exposure_summary.json"
    ),
    "cashout_analysis": (
        "reports/cashout_failure_analysis/"
        "cashout_failure_summary.json"
    ),
    "temporal_drift": (
        "reports/temporal_drift_audit/"
        "temporal_drift_summary.json"
    ),
    "formal_drift": (
        "reports/formal_feature_drift/"
        "formal_drift_metadata.json"
    ),
    "segment_drift": (
        "reports/segment_level_fraud_drift/"
        "segment_drift_summary.json"
    ),
    "grouped_shap": (
        "reports/grouped_shap_explainability/"
        "grouped_shap_summary.json"
    ),
    "robustness": (
        "reports/robustness_stress_test/"
        "robustness_summary.json"
    ),
    "coherent_stress": (
        "reports/coherent_financial_stress_test/"
        "coherent_stress_summary.json"
    ),
}


RISK_REGISTER = [
    {
        "risk_id": "MR-001",
        "category": "Data",
        "risk": "Synthetic dataset may not represent real banking traffic.",
        "evidence": "PaySim synthetic dataset; temporal and segment audits.",
        "likelihood": "High",
        "impact": "Critical",
        "priority": "Critical",
        "mitigation": (
            "Validate on representative, legally approved real transaction "
            "data before production use."
        ),
        "monitoring": (
            "Compare real transaction distributions, fraud prevalence, "
            "transaction types and account behaviour."
        ),
        "owner": "Data Science / Model Risk",
        "status": "Open",
    },
    {
        "risk_id": "MR-002",
        "category": "Performance",
        "risk": "CASH_OUT fraud detection is weak in the 50K-250K amount band.",
        "evidence": (
            "164 of 168 frauds in this test segment were missed; "
            "segment recall approximately 2.38%."
        ),
        "likelihood": "High",
        "impact": "Critical",
        "priority": "Critical",
        "mitigation": (
            "Investigate segment-specific features and collect representative "
            "real-world cases. Do not change the frozen threshold using test data."
        ),
        "monitoring": (
            "Track CASH_OUT recall, alert rate and missed-fraud exposure "
            "by amount band when labels become available."
        ),
        "owner": "Fraud Analytics",
        "status": "Open",
    },
    {
        "risk_id": "MR-003",
        "category": "Distribution Shift",
        "risk": "Fraud prevalence and transaction mix vary substantially over time.",
        "evidence": (
            "Train prevalence 0.0951%; validation 0.6173%; "
            "test 1.3994%."
        ),
        "likelihood": "High",
        "impact": "Critical",
        "priority": "Critical",
        "mitigation": (
            "Establish population, segment and label monitoring; "
            "review model validity when material changes occur."
        ),
        "monitoring": (
            "Track prevalence, transaction mix, feature PSI, "
            "segment-level performance and alert volumes."
        ),
        "owner": "Model Monitoring",
        "status": "Open",
    },
    {
        "risk_id": "MR-004",
        "category": "Financial Exposure",
        "risk": "False negatives may involve substantial transaction amounts.",
        "evidence": (
            "Neural model missed-fraud gross transaction amount was "
            "approximately $26.35M on the held-out synthetic test."
        ),
        "likelihood": "High",
        "impact": "Critical",
        "priority": "Critical",
        "mitigation": (
            "Use exposure-weighted review and investigate high-risk "
            "missed-fraud segments. Gross amount is not confirmed bank loss."
        ),
        "monitoring": (
            "Track missed-fraud count, gross amount, amount distribution "
            "and confirmed financial loss separately."
        ),
        "owner": "Fraud Operations / Risk",
        "status": "Open",
    },
    {
        "risk_id": "MR-005",
        "category": "Threshold Governance",
        "risk": "A frozen validation threshold may not remain suitable under changing prevalence.",
        "evidence": (
            "Frozen threshold 0.1444527954; validation and test "
            "prevalence differ materially."
        ),
        "likelihood": "High",
        "impact": "High",
        "priority": "High",
        "mitigation": (
            "Require formal approval, validation and versioning for any "
            "future threshold change. Never silently retune."
        ),
        "monitoring": (
            "Monitor alert rate, precision, recall and operational review capacity."
        ),
        "owner": "Model Owner / Risk Committee",
        "status": "Open",
    },
    {
        "risk_id": "MR-006",
        "category": "Explainability",
        "risk": "SHAP explanations describe model behaviour, not causal drivers.",
        "evidence": (
            "Grouped SHAP experiment; correlated numeric features can "
            "still produce unrealistic masked combinations."
        ),
        "likelihood": "Medium",
        "impact": "High",
        "priority": "High",
        "mitigation": (
            "Treat explanations as diagnostic evidence; validate with "
            "domain experts and avoid causal claims."
        ),
        "monitoring": (
            "Review explanation stability and feature attribution changes "
            "across relevant segments."
        ),
        "owner": "Data Science / Model Risk",
        "status": "Open",
    },
    {
        "risk_id": "MR-007",
        "category": "Input Integrity",
        "risk": "Invalid or malformed inputs may cause unsafe predictions.",
        "evidence": (
            "Six input validation rejection tests passed in robustness testing."
        ),
        "likelihood": "Medium",
        "impact": "High",
        "priority": "High",
        "mitigation": (
            "Enforce schema, finite numeric values, valid transaction type "
            "and feature ordering at every scoring boundary."
        ),
        "monitoring": (
            "Track rejected requests, schema failures, missing fields "
            "and invalid-value rates."
        ),
        "owner": "ML Engineering",
        "status": "Partially Mitigated",
    },
    {
        "risk_id": "MR-008",
        "category": "Accounting Consistency",
        "risk": "Raw transaction balance equations may not reconcile.",
        "evidence": (
            "Validation accounting audit found substantial residuals, "
            "especially for the origin-balance equation."
        ),
        "likelihood": "High",
        "impact": "High",
        "priority": "High",
        "mitigation": (
            "Define transaction-type-specific accounting rules and "
            "investigate exceptions before treating residuals as errors."
        ),
        "monitoring": (
            "Track equation residual distributions and reconciliation "
            "failure rates by transaction type."
        ),
        "owner": "Data Engineering / Fraud Analytics",
        "status": "Open",
    },
    {
        "risk_id": "MR-009",
        "category": "Operational",
        "risk": "No demonstrated production-grade monitoring and incident process yet.",
        "evidence": (
            "Offline experiments and reports exist; live monitoring, "
            "on-call ownership and incident drills are not established."
        ),
        "likelihood": "Medium",
        "impact": "Critical",
        "priority": "Critical",
        "mitigation": (
            "Build monitoring, alerting, runbooks, incident ownership "
            "and rollback procedures before a live pilot."
        ),
        "monitoring": (
            "Track service health, latency, failures, model version, "
            "data freshness and alert delivery."
        ),
        "owner": "ML Platform / Operations",
        "status": "Open",
    },
    {
        "risk_id": "MR-010",
        "category": "Privacy / Compliance",
        "risk": "Production banking data may have privacy, retention and access constraints.",
        "evidence": (
            "No production data governance or compliance approval has "
            "been established in this research project."
        ),
        "likelihood": "Unknown",
        "impact": "Critical",
        "priority": "Critical",
        "mitigation": (
            "Obtain legal, privacy, security and compliance review; "
            "minimise and protect sensitive data."
        ),
        "monitoring": (
            "Audit access, retention, encryption, data lineage and "
            "approved processing purposes."
        ),
        "owner": "Security / Compliance",
        "status": "Open",
    },
    {
        "risk_id": "MR-011",
        "category": "Model Reproducibility",
        "risk": "Model, feature transformation or threshold versions may become inconsistent.",
        "evidence": (
            "Frozen model hash verification and validation reconciliation "
            "are implemented in experiments."
        ),
        "likelihood": "Medium",
        "impact": "High",
        "priority": "High",
        "mitigation": (
            "Version model, feature schema, scaler, threshold and "
            "configuration as one approved release."
        ),
        "monitoring": (
            "Verify model hash and feature schema at startup and record "
            "versions with every prediction."
        ),
        "owner": "ML Engineering",
        "status": "Partially Mitigated",
    },
    {
        "risk_id": "MR-012",
        "category": "Human Oversight",
        "risk": "Automated decisions may cause inappropriate customer impact.",
        "evidence": (
            "Model produces REVIEW-style risk signals; production "
            "decision governance has not been established."
        ),
        "likelihood": "Medium",
        "impact": "Critical",
        "priority": "Critical",
        "mitigation": (
            "Use human review for consequential actions; establish "
            "appeal, override and escalation processes."
        ),
        "monitoring": (
            "Track reviewer overrides, confirmed outcomes, complaints "
            "and decision turnaround times."
        ),
        "owner": "Fraud Operations / Compliance",
        "status": "Open",
    },
]


READINESS = [
    {
        "dimension": "Offline model evaluation",
        "status": "Completed",
        "evidence": "Frozen temporal evaluation and benchmark reports.",
        "gap": "Results are based on synthetic PaySim data.",
    },
    {
        "dimension": "Segment-level risk analysis",
        "status": "Completed",
        "evidence": "CASH_OUT amount-band and missed-fraud investigations.",
        "gap": "Weak 50K-250K CASH_OUT segment requires remediation.",
    },
    {
        "dimension": "Explainability",
        "status": "Completed with limitations",
        "evidence": "Grouped SHAP experiment and additivity checks.",
        "gap": "Explanations are not causal and need domain validation.",
    },
    {
        "dimension": "Drift and robustness analysis",
        "status": "Research completed",
        "evidence": "Temporal, formal feature and stress testing reports.",
        "gap": "Live drift monitoring and alert thresholds are not calibrated.",
    },
    {
        "dimension": "Real-world data validation",
        "status": "Not completed",
        "evidence": "No representative real banking dataset.",
        "gap": "Required before any production performance claim.",
    },
    {
        "dimension": "Scoring service and integration",
        "status": "Not assessed in this governance phase",
        "evidence": "Requires a separately verified production scoring workflow.",
        "gap": "Authentication, resilience, load and integration tests.",
    },
    {
        "dimension": "Monitoring and incident response",
        "status": "Not completed",
        "evidence": "Offline reports only.",
        "gap": "Monitoring service, alert delivery, runbooks and rollback.",
    },
    {
        "dimension": "Security and compliance",
        "status": "Not completed",
        "evidence": "No formal approval evidence.",
        "gap": "Privacy, access control, security and compliance review.",
    },
    {
        "dimension": "Human decision governance",
        "status": "Not completed",
        "evidence": "No operational review workflow evidence.",
        "gap": "Human oversight, override and escalation process.",
    },
]


MONITORING_REQUIREMENTS = {
    "purpose": (
        "Proposed monitoring specification for a future controlled pilot. "
        "Not an approved production policy."
    ),
    "model_identity": {
        "frozen_model_sha256": (
            "ae9d3e357c2e0dcf9abe0600245eede7ebfc0198338009a5c084185f7302189b"
        ),
        "frozen_validation_threshold": 0.1444527953863144,
        "require_hash_verification": True,
        "record_model_version_per_prediction": True,
        "record_feature_schema_version": True,
    },
    "metrics": [
        {
            "name": "input_schema_failure_rate",
            "type": "data_quality",
            "frequency": "continuous",
            "action": "Investigate immediately if schema validation fails.",
        },
        {
            "name": "invalid_numeric_input_rate",
            "type": "data_quality",
            "frequency": "continuous",
            "action": "Reject invalid values; investigate source system.",
        },
        {
            "name": "transaction_type_mix",
            "type": "data_drift",
            "frequency": "daily",
            "action": "Compare with approved reference distribution.",
        },
        {
            "name": "feature_psi",
            "type": "data_drift",
            "frequency": "daily_or_weekly",
            "action": (
                "Use provisional PSI review bands: below 0.10 low, "
                "0.10-0.25 investigate, above 0.25 escalate. "
                "Calibrate these bands for real data."
            ),
        },
        {
            "name": "fraud_prevalence",
            "type": "label_drift",
            "frequency": "when_labels_arrive",
            "action": (
                "Compare observed prevalence with the approved reference; "
                "investigate material changes."
            ),
        },
        {
            "name": "precision_recall_pr_auc",
            "type": "model_performance",
            "frequency": "weekly_or_when_labels_arrive",
            "action": (
                "Calculate only on matured labels and compare by "
                "transaction type and amount band."
            ),
        },
        {
            "name": "cashout_50k_250k_recall",
            "type": "critical_segment_performance",
            "frequency": "weekly_or_when_labels_arrive",
            "action": (
                "Escalate poor recall; this segment is a known model weakness."
            ),
        },
        {
            "name": "alert_rate_and_queue_volume",
            "type": "operations",
            "frequency": "continuous_and_daily",
            "action": (
                "Compare with staffed review capacity; do not silently "
                "change the threshold."
            ),
        },
        {
            "name": "missed_fraud_gross_amount",
            "type": "financial_risk",
            "frequency": "when_labels_arrive",
            "action": (
                "Track gross transaction amount separately from "
                "confirmed realised loss."
            ),
        },
        {
            "name": "prediction_latency_and_error_rate",
            "type": "service_health",
            "frequency": "continuous",
            "action": "Alert on service degradation and failed scoring.",
        },
        {
            "name": "reviewer_override_rate",
            "type": "human_governance",
            "frequency": "weekly",
            "action": "Investigate repeated overrides and segment patterns.",
        },
    ],
    "governance_rules": [
        "Do not automatically retrain or retune from monitoring alerts.",
        "Require approval and validation for threshold changes.",
        "Keep an immutable prediction and model-version audit trail.",
        "Use matured fraud labels for performance calculations.",
        "Separate gross transaction exposure from confirmed financial loss.",
        "Escalate critical segment degradation to the model owner.",
    ],
    "important_limitation": (
        "Monitoring thresholds are proposed starting points, not "
        "empirically approved limits for a real financial institution."
    ),
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 72)
    print("MODEL RISK REGISTER AND DEPLOYMENT READINESS")
    print("=" * 72)

    evidence_status = {
        name: {
            "path": relative_path,
            "exists": (ROOT / relative_path).exists(),
        }
        for name, relative_path in EVIDENCE.items()
    }

    print("\nEvidence availability:")
    for name, item in evidence_status.items():
        print(
            f"  {'FOUND' if item['exists'] else 'MISSING'} "
            f"{name}: {item['path']}"
        )

    write_csv(
        OUTPUT / "model_risk_register.csv",
        RISK_REGISTER,
    )

    write_csv(
        OUTPUT / "deployment_readiness_scorecard.csv",
        READINESS,
    )

    (OUTPUT / "monitoring_requirements.json").write_text(
        json.dumps(MONITORING_REQUIREMENTS, indent=2),
        encoding="utf-8",
    )

    missing_evidence = [
        name
        for name, item in evidence_status.items()
        if not item["exists"]
    ]

    critical_risks = [
        risk for risk in RISK_REGISTER
        if risk["priority"] == "Critical"
    ]

    completed_dimensions = [
        item for item in READINESS
        if item["status"] == "Completed"
    ]

    decision = {
        "overall_decision": "CONDITIONAL_GO_FOR_SHADOW_ONLY",
        "autonomous_production_decision": "NO_GO",
        "human_reviewed_controlled_pilot": (
            "NOT_YET_APPROVED"
        ),
        "evidence_files_expected": len(EVIDENCE),
        "evidence_files_found": sum(
            item["exists"] for item in evidence_status.values()
        ),
        "missing_evidence": missing_evidence,
        "risk_count": len(RISK_REGISTER),
        "critical_risk_count": len(critical_risks),
        "readiness_dimensions": len(READINESS),
        "fully_completed_dimensions": len(completed_dimensions),
        "known_critical_model_weakness": (
            "CASH_OUT fraud between 50K and 250K: "
            "164 of 168 fraud cases were missed in the held-out "
            "synthetic test segment."
        ),
        "production_blockers": [
            "Representative real-world transaction and fraud-label validation.",
            "Remediation or formally accepted treatment of the weak CASH_OUT segment.",
            "Live monitoring, incident response and rollback procedures.",
            "Security, privacy and compliance approval.",
            "Human review, override and escalation governance.",
            "End-to-end service, integration and load testing.",
        ],
        "decision_rationale": (
            "Offline research evidence is substantial, but synthetic data, "
            "segment-level weaknesses and missing operational controls "
            "prevent autonomous production approval."
        ),
        "evidence_status": evidence_status,
    }

    (OUTPUT / "deployment_decision.json").write_text(
        json.dumps(decision, indent=2),
        encoding="utf-8",
    )

    markdown = f"""# Model Risk Management and Deployment Decision

## Decision

**{decision['overall_decision']}**

**Autonomous production decisions: {decision['autonomous_production_decision']}**

**Human-reviewed controlled pilot: {decision['human_reviewed_controlled_pilot']}**

This is a research-project governance assessment, not a formal approval
by a regulated financial institution.

## Evidence reviewed

- Evidence files found: {decision['evidence_files_found']} / {decision['evidence_files_expected']}
- Registered risks: {decision['risk_count']}
- Critical risks: {decision['critical_risk_count']}
- Readiness dimensions: {decision['readiness_dimensions']}

## Most important known weakness

{decision['known_critical_model_weakness']}

## Production blockers

"""

    for item in decision["production_blockers"]:
        markdown += f"- {item}\n"

    markdown += """
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
"""

    (OUTPUT / "deployment_decision.md").write_text(
        markdown,
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("GOVERNANCE ARTIFACTS CREATED")
    print("=" * 72)

    for path in sorted(OUTPUT.iterdir()):
        if path.is_file():
            print(f"  {path.name}")

    print("\nDecision:")
    print(f"  {decision['overall_decision']}")
    print(f"  Autonomous production: {decision['autonomous_production_decision']}")
    print(f"  Critical risks: {decision['critical_risk_count']}")
    print(f"  Evidence found: {decision['evidence_files_found']}/{decision['evidence_files_expected']}")

    if missing_evidence:
        print("\nWARNING: Some expected evidence files are missing.")
        print("Review deployment_decision.json before using this assessment.")


if __name__ == "__main__":
    main()
