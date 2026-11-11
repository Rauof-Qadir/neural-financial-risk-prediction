from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
API_URL = os.getenv("RISK_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Neural Risk Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# THEME
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #080e1a;
        --panel: #101a2b;
        --panel2: #142238;
        --line: rgba(148, 163, 184, .16);
        --muted: #91a2ba;
        --text: #edf4ff;
        --cyan: #50d8e8;
        --green: #36d399;
        --amber: #f6bd60;
        --red: #fb7185;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
          radial-gradient(ellipse at 10% 0%, rgba(39, 91, 143, .18), transparent 35%),
          radial-gradient(ellipse at 90% 10%, rgba(29, 120, 132, .10), transparent 30%),
          var(--bg);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: #0b1423;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * {
        color: #dce8f8;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -.035em;
    }

    h1 { font-size: 2.2rem !important; }
    h2 { font-size: 1.45rem !important; }
    h3 { font-size: 1.08rem !important; }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #142238, #101a2b);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 17px 18px;
        min-height: 118px;
    }

    [data-testid="stMetricLabel"] {
        color: #a8b8cd !important;
        font-size: .83rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #f2f7ff !important;
        font-family: 'Space Grotesk', sans-serif;
    }

    [data-testid="stMetricDelta"] { font-size: .76rem; }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important;
        border-radius: 15px !important;
    }

    .hero {
        background: linear-gradient(115deg, #142a43 0%, #102239 55%, #12313d 100%);
        border: 1px solid rgba(80, 216, 232, .22);
        border-radius: 19px;
        padding: 26px 30px;
        margin: 5px 0 24px 0;
    }

    .eyebrow {
        color: var(--cyan);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
    }

    .hero-title {
        color: #f4f8ff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        margin: 7px 0 5px 0;
    }

    .hero-sub {
        color: #b6c8dc;
        font-size: .94rem;
        max-width: 850px;
        line-height: 1.65;
    }

    .status-pill {
        display: inline-block;
        border: 1px solid rgba(246, 189, 96, .4);
        color: #ffd58b;
        background: rgba(246, 189, 96, .10);
        border-radius: 30px;
        padding: 6px 11px;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .06em;
    }

    .section-kicker {
        color: var(--cyan);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .muted { color: var(--muted); }
    .small-note { color: #9fb0c5; font-size: .82rem; line-height: 1.55; }

    .risk-card {
        background: #111d30;
        border: 1px solid var(--line);
        border-radius: 13px;
        padding: 17px;
        min-height: 110px;
    }

    .risk-card-title {
        color: #9fb0c5;
        font-size: .76rem;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    .risk-card-value {
        color: #f4f8ff;
        font-size: 1.45rem;
        font-weight: 700;
        font-family: 'Space Grotesk', sans-serif;
        margin-top: 7px;
    }

    .stButton > button, .stFormSubmitButton > button {
        border-radius: 9px;
        border: 1px solid rgba(80, 216, 232, .35);
        background: #123044;
        color: #e9fbff;
        font-weight: 600;
    }

    .stButton > button:hover, .stFormSubmitButton > button:hover {
        border-color: var(--cyan);
        color: white;
        background: #17445a;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 10px;
        overflow: hidden;
    }

    hr { border-color: var(--line) !important; }

    #MainMenu, footer { visibility: hidden; }

    @media (max-width: 800px) {
        .hero { padding: 20px; }
        .hero-title { font-size: 1.5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# DATA ACCESS
# ---------------------------------------------------------------------

def read_json(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def read_csv(relative_path: str) -> pd.DataFrame:
    path = ROOT / relative_path
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_static_reports() -> dict[str, Any]:
    return {
        "final": read_json("reports/final_evaluation/final_test_results.json"),
        "diagnostics": read_json("reports/diagnostics/model_diagnostics.json"),
        "neural_experiment": read_json("reports/experiments/neural_network_results.json"),
        "baseline_experiment": read_json("reports/experiments/baseline_v2_results.json"),
        "business": read_json("reports/business_impact/business_impact_report.json"),
        "governance": read_json("reports/model_governance/deployment_decision.json"),
        "shap": read_json("reports/grouped_shap_explainability/grouped_shap_summary.json"),
        "temporal": read_json("reports/temporal_drift_audit/temporal_drift_summary.json"),
        "coherent_stress": read_json("reports/coherent_financial_stress_test/coherent_stress_summary.json"),
        "robustness": read_json("reports/robustness_stress_test/robustness_summary.json"),
        "missed": read_json("reports/missed_fraud_exposure/missed_fraud_exposure_summary.json"),
        "cashout": read_json("reports/cashout_failure_analysis/cashout_failure_summary.json"),
        "segment": read_json("reports/segment_level_fraud_drift/segment_drift_summary.json"),
        "formal_drift": read_json("reports/formal_feature_drift/formal_drift_metadata.json"),
    }


def find_report_csv(patterns: list[str]) -> pd.DataFrame:
    for pattern in patterns:
        matches = list(REPORTS.rglob(pattern))
        for path in matches:
            try:
                frame = pd.read_csv(path)
                if not frame.empty:
                    return frame
            except Exception:
                continue
    return pd.DataFrame()


def api_get(endpoint: str) -> tuple[bool, Any]:
    try:
        response = httpx.get(f"{API_URL}{endpoint}", timeout=3.0)
        response.raise_for_status()
        return True, response.json()
    except Exception as exc:
        return False, str(exc)


def api_post(endpoint: str, payload: dict[str, Any]) -> tuple[bool, Any]:
    try:
        response = httpx.post(
            f"{API_URL}{endpoint}",
            json=payload,
            timeout=10.0,
        )
        response.raise_for_status()
        return True, response.json()
    except Exception as exc:
        return False, str(exc)


def pct(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def money(value: Any) -> str:
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "—"


def show_missing(message: str) -> None:
    st.info(message)


def page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="muted">{description}</p>', unsafe_allow_html=True)


def plot_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#dce8f8"),
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(148,163,184,.10)", zerolinecolor="rgba(148,163,184,.10)"),
        yaxis=dict(gridcolor="rgba(148,163,184,.10)", zerolinecolor="rgba(148,163,184,.10)"),
    )
    return fig


DATA = load_static_reports()
FINAL = DATA["final"]
TEST = FINAL.get("test_data", {})
NN = FINAL.get("neural_network", {}).get("results", {})
LR = FINAL.get("logistic_regression", {}).get("results", {})
GOV = DATA["governance"]
BUSINESS = DATA["business"]


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="padding:12px 2px 18px 2px">
          <div style="font-size:1.45rem;font-weight:700;font-family:Space Grotesk;color:#f3f8ff">
            ◈ RISK<span style="color:#50d8e8">INTEL</span>
          </div>
          <div style="color:#8fa4bd;font-size:.72rem;letter-spacing:.12em;margin-top:4px">
            FINANCIAL MODEL ANALYTICS
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "NAVIGATION",
        [
            "Executive Overview",
            "Model Comparison",
            "Fraud Intelligence",
            "Explainability",
            "Drift & Stress",
            "Business Impact",
            "Live Risk Scoring",
            "Prediction Monitoring",
            "Model Governance",
        ],
        label_visibility="visible",
    )

    st.divider()
    st.markdown("**SYSTEM STATUS**")
    st.markdown('<span class="status-pill">SHADOW MODE ONLY</span>', unsafe_allow_html=True)
    st.caption("Synthetic PaySim research evaluation")
    st.caption("Model: frozen PyTorch MLP")
    st.caption(f"API: {API_URL}")

    if st.button("Refresh report data", width='stretch'):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("Neural Financial Risk Prediction System")
    st.caption("Research dashboard · Not a production banking system")


# ---------------------------------------------------------------------
# EXECUTIVE OVERVIEW
# ---------------------------------------------------------------------

if page == "Executive Overview":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Financial risk intelligence platform</div>
          <div class="hero-title">Neural Financial Risk Prediction</div>
          <div class="hero-sub">
            An evidence-led comparison of a PyTorch neural network and a
            conventional Logistic Regression baseline, evaluated on an
            untouched temporal test set from the synthetic PaySim dataset.
          </div>
          <div style="margin-top:17px">
            <span class="status-pill">CONDITIONAL GO · SHADOW ONLY</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not NN or not LR:
        show_missing("Final evaluation report is missing. Expected: reports/final_evaluation/final_test_results.json")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Test transactions", f"{TEST.get('rows', 0):,}")
        c2.metric("Fraud cases", f"{TEST.get('fraud_transactions', 0):,}")
        c3.metric("Neural PR-AUC", number(NN.get("pr_auc")))
        c4.metric("Neural precision", pct(NN.get("precision")))
        c5.metric("Neural recall", pct(NN.get("recall")))

        st.markdown("### Model outcome")
        left, right = st.columns([1.35, 1])

        with left:
            comparison = pd.DataFrame(
                {
                    "Metric": ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1"],
                    "Neural Network": [
                        NN.get("pr_auc"), NN.get("roc_auc"), NN.get("precision"),
                        NN.get("recall"), NN.get("f1")
                    ],
                    "Logistic Regression": [
                        LR.get("pr_auc"), LR.get("roc_auc"), LR.get("precision"),
                        LR.get("recall"), LR.get("f1")
                    ],
                }
            )
            fig = px.bar(
                comparison.melt(id_vars="Metric", var_name="Model", value_name="Score"),
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                title="Held-out temporal test performance",
                color_discrete_sequence=["#50d8e8", "#8496b2"],
            )
            st.plotly_chart(plot_layout(fig, 390), width='stretch')

        with right:
            cm = NN.get("confusion_matrix", {})
            st.markdown("**Neural network confusion matrix**")
            matrix = pd.DataFrame(
                [
                    [cm.get("true_negative", 0), cm.get("false_positive", 0)],
                    [cm.get("false_negative", 0), cm.get("true_positive", 0)],
                ],
                index=["Actual normal", "Actual fraud"],
                columns=["Predicted normal", "Predicted review"],
            )
            fig = px.imshow(
                matrix,
                text_auto=",",
                aspect="auto",
                color_continuous_scale="Teal",
                title="Neural network · frozen threshold",
            )
            st.plotly_chart(plot_layout(fig, 300), width='stretch')

        st.markdown("### Operational snapshot")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Neural alerts", f"{NN.get('alert_count', 0):,}")
        c2.metric("Baseline alerts", f"{LR.get('alert_count', 0):,}")
        c3.metric("Neural alert rate", pct(NN.get("alert_rate")))
        c4.metric("Baseline alert rate", pct(LR.get("alert_rate")))

        st.warning(
            "Critical limitation: the neural model missed 164 of 168 fraud cases "
            "in the $50K–$250K CASH_OUT test segment. Overall metrics must not "
            "hide this segment-level weakness."
        )

        st.markdown("### Evaluation integrity")
        st.write(
            "Both decision thresholds were selected on validation data and frozen "
            "before the final temporal test. No test-based threshold tuning was performed."
        )
        st.caption(
            f"Test prevalence: {pct(TEST.get('fraud_prevalence'))} · "
            f"Validation prevalence: {pct(FINAL.get('distribution_comparison', {}).get('validation_fraud_prevalence'))} · "
            f"Test/validation ratio: {number(FINAL.get('distribution_comparison', {}).get('test_to_validation_prevalence_ratio'), 2)}×"
        )


# ---------------------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------------------

elif page == "Model Comparison":
    page_header(
        "PERFORMANCE LAB",
        "Model Comparison",
        "Compare predictive quality, alert workload, validation behavior and model training characteristics.",
    )

    if not NN or not LR:
        show_missing("Final model evaluation JSON is unavailable.")
    else:
        rows = []
        for label, values in [("PyTorch MLP", NN), ("Logistic Regression", LR)]:
            rows.append(
                {
                    "Model": label,
                    "PR-AUC": values.get("pr_auc"),
                    "ROC-AUC": values.get("roc_auc"),
                    "Precision": values.get("precision"),
                    "Recall": values.get("recall"),
                    "F1": values.get("f1"),
                    "Alert rate": values.get("alert_rate"),
                    "Alerts": values.get("alert_count"),
                    "Brier score": values.get("brier_score"),
                    "Log loss": values.get("log_loss"),
                    "Threshold": values.get("threshold"),
                }
            )

        table = pd.DataFrame(rows)
        c1, c2 = st.columns(2)
        c1.metric("Neural PR-AUC", number(NN.get("pr_auc")))
        c2.metric("Baseline PR-AUC", number(LR.get("pr_auc")))
        st.dataframe(
            table.style.format(
                {
                    "PR-AUC": "{:.4f}", "ROC-AUC": "{:.4f}",
                    "Precision": "{:.2%}", "Recall": "{:.2%}",
                    "F1": "{:.4f}", "Alert rate": "{:.2%}",
                    "Brier score": "{:.5f}", "Log loss": "{:.5f}",
                    "Threshold": "{:.6f}",
                },
                na_rep="—",
            ),
            width='stretch',
            hide_index=True,
        )

        st.markdown("### Alert workload")
        workload = pd.DataFrame(
            {
                "Model": ["PyTorch MLP", "Logistic Regression"],
                "Alerts": [NN.get("alert_count", 0), LR.get("alert_count", 0)],
            }
        )
        fig = px.bar(
            workload,
            x="Model",
            y="Alerts",
            text="Alerts",
            color="Model",
            color_discrete_sequence=["#50d8e8", "#8496b2"],
            title="Number of transactions sent for review",
        )
        st.plotly_chart(plot_layout(fig, 330), width='stretch')

        neural_run = DATA["neural_experiment"]
        baseline_run = DATA["baseline_experiment"]
        if neural_run and baseline_run:
            st.markdown("### Training profile")
            t1, t2, t3 = st.columns(3)
            t1.metric("MLP training time", f"{neural_run.get('training_seconds', 0):.1f} sec")
            t2.metric("Baseline training time", f"{baseline_run.get('training_seconds', 0):.1f} sec")
            t3.metric("Best neural epoch", neural_run.get("best_epoch", "—"))

            history = neural_run.get("history", [])
            if history:
                history_df = pd.DataFrame(history)
                fig = px.line(
                    history_df,
                    x="epoch",
                    y="validation_pr_auc",
                    markers=True,
                    title="Neural validation PR-AUC by epoch",
                    labels={"validation_pr_auc": "Validation PR-AUC", "epoch": "Epoch"},
                )
                st.plotly_chart(plot_layout(fig, 320), width='stretch')

        st.info(
            "Latency values in the diagnostics report are model-only batch benchmarks "
            "on CPU (batch size 1,024), not end-to-end API response times. "
            "P99 is based on only 50 benchmark repetitions."
        )


# ---------------------------------------------------------------------
# FRAUD INTELLIGENCE
# ---------------------------------------------------------------------

elif page == "Fraud Intelligence":
    page_header(
        "SEGMENT ANALYTICS",
        "Fraud Intelligence",
        "Inspect error patterns and understand where the frozen neural model struggles.",
    )

    cashout = DATA["cashout"]
    missed = DATA["missed"]

    cm = NN.get("confusion_matrix", {})
    a, b, c, d = st.columns(4)
    a.metric("False positives", f"{cm.get('false_positive', 0):,}")
    b.metric("False negatives", f"{cm.get('false_negative', 0):,}")
    c.metric("True positives", f"{cm.get('true_positive', 0):,}")
    d.metric("True negatives", f"{cm.get('true_negative', 0):,}")

    st.markdown("### Critical CASH_OUT segment")
    st.error(
        "164 of 168 fraud transactions between $50,000 and $250,000 in the "
        "CASH_OUT test segment were missed by the neural model."
    )

    if cashout:
        st.json(cashout, expanded=False)
    else:
        st.caption("Detailed cash-out report not found at its expected path.")

    if missed:
        st.markdown("### Missed fraud exposure summary")
        st.json(missed, expanded=False)

    st.markdown("### Interpretation")
    st.write(
        "This is a descriptive error pattern, not proof of a causal mechanism. "
        "The model's aggregate performance does not imply uniform performance "
        "across transaction types or amount ranges."
    )
    st.caption(
        "Gross missed transaction amount is an exposure proxy. It is not verified "
        "realized loss, recoverable loss, or a direct estimate of bank loss."
    )


# ---------------------------------------------------------------------
# EXPLAINABILITY
# ---------------------------------------------------------------------

elif page == "Explainability":
    page_header(
        "MODEL INTERPRETABILITY",
        "Explainability",
        "Grouped SHAP analysis of the frozen neural model.",
    )

    shap_summary = DATA["shap"]
    if shap_summary:
        st.markdown("### Grouped SHAP summary")
        st.json(shap_summary, expanded=True)
    else:
        st.info(
            "SHAP summary JSON was not found. Expected: "
            "reports/grouped_shap_explainability/grouped_shap_summary.json"
        )

    shap_csv = find_report_csv(
        [
            "*global*importance*.csv",
            "*shap*importance*.csv",
            "*feature*importance*.csv",
        ]
    )
    if not shap_csv.empty:
        st.markdown("### Feature importance data")
        st.dataframe(shap_csv, width='stretch', hide_index=True)

    plot_dir = REPORTS / "grouped_shap_explainability" / "plots"
    image_files = []
    if plot_dir.exists():
        image_files = sorted(
            list(plot_dir.glob("*.png")) + list(plot_dir.glob("*.jpg"))
        )

    if image_files:
        st.markdown("### Generated explainability plots")
        for image_path in image_files[:6]:
            st.image(str(image_path), caption=image_path.stem, width='stretch')
    else:
        st.caption("No image plots found in the grouped SHAP plots directory.")

    st.warning(
        "SHAP values describe model behavior, not causality. Independently masking "
        "numeric features may create unrealistic transaction combinations."
    )


# ---------------------------------------------------------------------
# DRIFT & STRESS
# ---------------------------------------------------------------------

elif page == "Drift & Stress":
    page_header(
        "ROBUSTNESS & DISTRIBUTION SHIFT",
        "Drift & Stress Testing",
        "Review temporal prevalence changes, feature drift evidence and stress-test summaries.",
    )

    temporal = DATA["temporal"]
    if temporal:
        st.markdown("### Temporal drift audit")
        st.json(temporal, expanded=True)
    else:
        st.info("Temporal drift summary was not found.")

    drift_csv = find_report_csv(
        [
            "*numeric_feature_drift*.csv",
            "*feature_drift*.csv",
            "*categorical*drift*.csv",
        ]
    )
    if not drift_csv.empty:
        st.markdown("### Feature drift records")
        st.dataframe(drift_csv.head(300), width='stretch', hide_index=True)
        if "feature" in drift_csv.columns and "psi" in drift_csv.columns:
            top = drift_csv.sort_values("psi", ascending=False).head(15)
            fig = px.bar(
                top,
                x="psi",
                y="feature",
                color="population" if "population" in top.columns else None,
                orientation="h",
                title="Largest reported PSI values",
            )
            st.plotly_chart(plot_layout(fig, 430), width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Robustness stress test")
        if DATA["robustness"]:
            st.json(DATA["robustness"], expanded=False)
        else:
            st.caption("Robustness summary not found.")

    with c2:
        st.markdown("### Coherent financial stress test")
        if DATA["coherent_stress"]:
            st.json(DATA["coherent_stress"], expanded=False)
        else:
            st.caption("Coherent stress summary not found.")

    if DATA["formal_drift"]:
        st.markdown("### Formal feature drift metadata")
        st.json(DATA["formal_drift"], expanded=False)

    st.warning(
        "Temporal splits intentionally cover different time ranges. Large step-feature "
        "PSI is therefore expected and is not, by itself, proof of harmful concept drift. "
        "Synthetic stress tests are not real-world performance guarantees."
    )


# ---------------------------------------------------------------------
# BUSINESS IMPACT
# ---------------------------------------------------------------------

elif page == "Business Impact":
    page_header(
        "COST & REVIEW WORKLOAD",
        "Business Impact",
        "Illustrative operational-cost scenarios using the held-out test confusion matrices.",
    )

    if not BUSINESS:
        show_missing("Business impact report not found.")
    else:
        assumptions = BUSINESS.get("assumptions", {})
        scenarios = BUSINESS.get("scenario_comparison", [])
        models = BUSINESS.get("models", {})

        st.warning(
            "These are illustrative calculations, not actual bank costs or realized savings. "
            "The assumed false-positive review costs and false-negative costs must be "
            "replaced with institution-approved estimates."
        )

        if scenarios:
            scenario_df = pd.DataFrame(scenarios)
            fig = px.bar(
                scenario_df,
                x="scenario",
                y=["baseline_estimated_cost", "neural_estimated_cost"],
                barmode="group",
                title="Estimated operational cost by scenario",
                labels={"value": "Estimated cost (USD)", "variable": "Model"},
                color_discrete_sequence=["#8496b2", "#50d8e8"],
            )
            st.plotly_chart(plot_layout(fig, 370), width='stretch')

            st.dataframe(
                scenario_df.style.format(
                    {
                        "baseline_estimated_cost": "${:,.0f}",
                        "neural_estimated_cost": "${:,.0f}",
                        "estimated_savings": "${:,.0f}",
                        "savings_percentage": "{:.2f}%",
                    }
                ),
                width='stretch',
                hide_index=True,
            )

        base = next((s for s in scenarios if s.get("scenario") == "base"), {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Illustrative baseline cost", money(base.get("baseline_estimated_cost")))
        c2.metric("Illustrative neural cost", money(base.get("neural_estimated_cost")))
        c3.metric("Illustrative difference", money(base.get("estimated_savings")))

        st.markdown("### False-positive and missed-fraud transaction amounts")
        if models:
            rows = []
            for key, label in [
                ("neural_network", "PyTorch MLP"),
                ("logistic_regression", "Logistic Regression"),
            ]:
                item = models.get(key, {})
                rows.append(
                    {
                        "Model": label,
                        "False positive amount": item.get("false_positive_transaction_amount"),
                        "Missed fraud amount": item.get("missed_fraud_transaction_amount"),
                        "Missed fraud count": item.get("missed_fraud_count"),
                    }
                )
            st.dataframe(
                pd.DataFrame(rows).style.format(
                    {
                        "False positive amount": "${:,.2f}",
                        "Missed fraud amount": "${:,.2f}",
                    }
                ),
                width='stretch',
                hide_index=True,
            )

        st.caption(
            "Missed-fraud transaction amount is gross transaction exposure, not verified "
            "financial loss. It must not be interpreted as money definitively lost by a bank."
        )


# ---------------------------------------------------------------------
# LIVE RISK SCORING
# ---------------------------------------------------------------------

elif page == "Live Risk Scoring":
    page_header(
        "INTERACTIVE MODEL INFERENCE",
        "Live Risk Scoring",
        "Submit a sample transaction to the frozen API. Scoring is advisory and shadow-only.",
    )

    ok, health = api_get("/health")
    if ok:
        st.success(
            f"API connected · Status: {health.get('status')} · "
            f"Deployment: {health.get('deployment_mode')}"
        )
    else:
        st.error("API is unavailable. Start the FastAPI service to enable live scoring.")
        st.code("uvicorn src.api.main:app --reload", language="powershell")
        st.caption(f"Connection detail: {health}")

    model_ok, model_info = api_get("/model-info")
    if model_ok:
        with st.expander("Loaded model details"):
            st.json(model_info)

    st.markdown("### Transaction input")
    with st.form("risk_scoring_form"):
        col1, col2 = st.columns(2)
        with col1:
            step = st.number_input("Time step", min_value=1, value=650, step=1)
            transaction_type = st.selectbox(
                "Transaction type",
                ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"],
            )
            amount = st.number_input("Transaction amount", min_value=0.0, value=150000.0, step=1000.0)
        with col2:
            oldbalance_org = st.number_input("Origin old balance", min_value=0.0, value=150000.0, step=1000.0)
            oldbalance_dest = st.number_input("Destination old balance", min_value=0.0, value=50000.0, step=1000.0)

        submitted = st.form_submit_button("Score transaction", width='stretch')

    if submitted:
        payload = {
            "step": int(step),
            "type": transaction_type,
            "amount": float(amount),
            "oldbalanceOrg": float(oldbalance_org),
            "oldbalanceDest": float(oldbalance_dest),
        }
        success, result = api_post("/predict", payload)
        if success:
            probability = float(result.get("risk_probability", 0))
            decision = result.get("risk_decision", "UNKNOWN")
            st.markdown("### Scoring result")
            a, b, c = st.columns(3)
            a.metric("Risk probability", pct(probability, 3))
            b.metric("Decision", decision)
            c.metric("Frozen threshold", pct(result.get("threshold"), 3))

            if decision == "REVIEW":
                st.warning("This transaction crossed the model threshold and is marked for human review.")
            else:
                st.success("This transaction is below the model review threshold.")

            st.info(
                f"Deployment mode: {result.get('deployment_mode')} · "
                f"Shadow only: {result.get('shadow_only')}"
            )
            with st.expander("Full API response"):
                st.json(result)
        else:
            st.error(f"Scoring request failed: {result}")

    st.caption(
        "This interface does not approve, decline, block, or execute transactions. "
        "Do not enter real customer-identifying information."
    )


# ---------------------------------------------------------------------
# PREDICTION MONITORING
# ---------------------------------------------------------------------

elif page == "Prediction Monitoring":
    page_header(
        "MODEL OPERATIONS",
        "Prediction Monitoring",
        "Inspect local scoring events and feature-distribution drift from the monitoring API.",
    )

    summary_ok, summary = api_get("/monitoring/summary")
    drift_ok, drift = api_get("/monitoring/drift")

    if not summary_ok and not drift_ok:
        st.warning("Monitoring API is unavailable. Start the FastAPI service to inspect current monitoring data.")
        st.code("uvicorn src.api.main:app --reload", language="powershell")
    else:
        if summary_ok:
            st.markdown("### Prediction summary")
            st.json(summary, expanded=True)
        if drift_ok:
            st.markdown("### Feature drift status")
            st.json(drift, expanded=True)

    monitoring_dir = REPORTS / "monitoring"
    if monitoring_dir.exists():
        st.markdown("### Saved monitoring artifacts")
        files = sorted(monitoring_dir.glob("*"))
        if files:
            st.dataframe(
                pd.DataFrame(
                    [{"Artifact": f.name, "Size (KB)": round(f.stat().st_size / 1024, 2)} for f in files if f.is_file()]
                ),
                width='stretch',
                hide_index=True,
            )

    st.info(
        "The monitoring implementation uses a minimum sample-size safeguard for PSI. "
        "Small samples may return INSUFFICIENT_DATA rather than a misleading drift estimate."
    )
    st.caption(
        "This is local demonstration monitoring. It is not an immutable audit system, "
        "does not provide enterprise retention controls, and is not a production alerting stack."
    )


# ---------------------------------------------------------------------
# MODEL GOVERNANCE
# ---------------------------------------------------------------------

elif page == "Model Governance":
    page_header(
        "MODEL RISK MANAGEMENT",
        "Model Governance",
        "Deployment readiness, known limitations, unresolved risks and evidence status.",
    )

    if not GOV:
        show_missing("Governance decision file not found.")
    else:
        decision = GOV.get("overall_decision", "UNKNOWN")
        st.markdown(
            f"""
            <div class="hero">
              <div class="eyebrow">Current governance decision</div>
              <div class="hero-title" style="font-size:1.65rem">{decision}</div>
              <div class="hero-sub">
                Autonomous production: {GOV.get('autonomous_production_decision', 'UNKNOWN')}<br>
                Human-reviewed pilot: {GOV.get('human_reviewed_controlled_pilot', 'UNKNOWN')}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Total risks", GOV.get("risk_count", "—"))
        c2.metric("Critical risks", GOV.get("critical_risk_count", "—"))
        c3.metric(
            "Evidence files",
            f"{GOV.get('evidence_files_found', 0)}/{GOV.get('evidence_files_expected', 0)}",
        )

        st.markdown("### Production blockers")
        for item in GOV.get("production_blockers", []):
            st.markdown(f"- {item}")

        st.markdown("### Known critical model weakness")
        st.error(GOV.get("known_critical_model_weakness", "Not specified."))

        st.markdown("### Decision rationale")
        st.write(GOV.get("decision_rationale", "No rationale available."))

        with st.expander("Full governance decision JSON"):
            st.json(GOV, expanded=True)

    st.markdown("### Deployment principles")
    st.markdown(
        """
        - No autonomous transaction approval, decline, or blocking.
        - Human review is required for risk alerts.
        - The model threshold remains frozen.
        - Final test data is not available to the scoring API.
        - Real-world representative data validation is still required.
        - Security, privacy, compliance, incident response and rollback controls
          require further work before any production consideration.
        """
    )


# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------

st.divider()
st.markdown(
    """
    <div style="text-align:center;color:#8194ad;font-size:.76rem;padding:8px 0 18px 0">
      NEURAL FINANCIAL RISK PREDICTION SYSTEM · RESEARCH / PORTFOLIO DASHBOARD<br>
      Synthetic PaySim data · Frozen temporal evaluation · Shadow-only scoring
    </div>
    """,
    unsafe_allow_html=True,
)

