"""
=============================================================
  Credit Card Default Analysis System – Streamlit Web App
=============================================================
  Run: streamlit run app.py
=============================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Default Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem;
        text-align: center; color: white;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; }
    .main-header p  { color: #a0aec0; margin-top: 0.5rem; }
    .metric-card {
        background: white; border-radius: 10px; padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
        border-top: 4px solid;
    }
    .risk-low    { border-color: #48bb78; }
    .risk-medium { border-color: #f6ad55; }
    .risk-high   { border-color: #fc8181; }
    .rec-box {
        background: #f7fafc; border-left: 4px solid #3182ce;
        padding: 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
    }
    .section-title { font-size: 1.3rem; font-weight: 700; margin: 1.5rem 0 1rem; }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model_path  = "models/random_forest_model.pkl"
    scaler_path = "models/scaler.pkl"
    features_path = "models/feature_names.pkl"
    if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
        return None, None, None
    model    = joblib.load(model_path)
    scaler   = joblib.load(scaler_path)
    features = joblib.load(features_path)
    return model, scaler, features

model, scaler, feature_names = load_artifacts()

# ─── Helper Functions ───────────────────────────────────────
def compute_risk_score(prob):
    return int(prob * 100)

def get_risk_level(score):
    if score <= 30:
        return "LOW", "#48bb78", "🟢"
    elif score <= 70:
        return "MEDIUM", "#f6ad55", "🟡"
    else:
        return "HIGH", "#fc8181", "🔴"

def get_recommendations(score, limit_bal, pay_delay, age):
    if score >= 71:
        return [
            "🔴 URGENT: Flag account for immediate risk review.",
            "💳 Suspend credit limit increases and block cash advances.",
            "📞 Assign a dedicated relationship manager.",
            "📋 Offer a structured EMI repayment plan (6–12 months).",
            "📧 Send overdue payment alert via SMS, email & push notification.",
            "📊 Monitor transaction patterns weekly.",
        ]
    elif score >= 31:
        return [
            "🟡 Reduce credit limit by 20–30% as a precaution.",
            "📩 Send friendly payment reminder 10 days before due date.",
            "💡 Offer cashback incentives for on-time payments.",
            "🤝 Offer payment restructuring if customer requests.",
            "📊 Review account monthly for changes in payment behavior.",
        ]
    else:
        return [
            "🟢 Account is in excellent standing — no action needed.",
            "🎁 Consider a credit limit increase as a loyalty reward.",
            "📈 Flag as premium upsell candidate (travel card / personal loan).",
            "⭐ Enroll in VIP rewards program.",
        ]

def make_gauge(score, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Risk Score", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color},
            "steps": [
                {"range": [0, 30],  "color": "#c6f6d5"},
                {"range": [30, 70], "color": "#fef3c7"},
                {"range": [70, 100],"color": "#fed7d7"},
            ],
            "threshold": {"line": {"color": "black", "width": 4}, "value": score},
        }
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

# ─── Header ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>💳 Credit Card Default Prediction System</h1>
    <p>AI-powered risk assessment for smarter credit decisions</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar – Input Form ───────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-cards.png", width=80)
    st.title("📝 Customer Input")
    st.markdown("---")

    st.subheader("👤 Demographics")
    age       = st.slider("Age", 20, 80, 35)
    sex       = st.selectbox("Gender", ["Male (1)", "Female (2)"])
    sex_val   = 1 if "Male" in sex else 2
    education = st.selectbox("Education",
                             ["Graduate School (1)", "University (2)", "High School (3)", "Other (4)"])
    edu_val   = int(education.split("(")[1][0])
    marriage  = st.selectbox("Marital Status", ["Married (1)", "Single (2)", "Other (3)"])
    mar_val   = int(marriage.split("(")[1][0])

    st.subheader("💰 Credit Information")
    limit_bal = st.number_input("Credit Limit (NTD)", 10000, 1000000, 50000, step=5000)

    st.subheader("💸 Bill Amounts (NTD)")
    bill1 = st.number_input("Bill Amount – Month 1", 0, 1000000, 15000, step=1000)
    bill2 = st.number_input("Bill Amount – Month 2", 0, 1000000, 14000, step=1000)
    bill3 = st.number_input("Bill Amount – Month 3", 0, 1000000, 12000, step=1000)
    bill4 = st.number_input("Bill Amount – Month 4", 0, 1000000, 11000, step=1000)
    bill5 = st.number_input("Bill Amount – Month 5", 0, 1000000, 10000, step=1000)
    bill6 = st.number_input("Bill Amount – Month 6", 0, 1000000, 9000,  step=1000)

    st.subheader("💵 Payment Amounts (NTD)")
    pay_amt1 = st.number_input("Payment – Month 1", 0, 500000, 3000, step=500)
    pay_amt2 = st.number_input("Payment – Month 2", 0, 500000, 2800, step=500)
    pay_amt3 = st.number_input("Payment – Month 3", 0, 500000, 2500, step=500)
    pay_amt4 = st.number_input("Payment – Month 4", 0, 500000, 2000, step=500)
    pay_amt5 = st.number_input("Payment – Month 5", 0, 500000, 1500, step=500)
    pay_amt6 = st.number_input("Payment – Month 6", 0, 500000, 1000, step=500)

    st.subheader("⏰ Payment Delay History")
    st.caption("(-2 = No credit used, -1 = Paid in full, 0 = Min payment, 1–9 = months delayed)")
    pay1 = st.slider("Delay Month 1 (Most Recent)", -2, 8, 0)
    pay2 = st.slider("Delay Month 2", -2, 8, 0)
    pay3 = st.slider("Delay Month 3", -2, 8, 0)
    pay4 = st.slider("Delay Month 4", -2, 8, 0)
    pay5 = st.slider("Delay Month 5", -2, 8, 0)
    pay6 = st.slider("Delay Month 6", -2, 8, 0)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Default Risk", use_container_width=True, type="primary")

# ─── Main Content ───────────────────────────────────────────
if not predict_btn:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📌 How to Use**\n\nFill in the customer details in the sidebar and click **Predict Default Risk**.")
    with col2:
        st.success("**🎯 What You'll Get**\n\nRisk score (0–100), risk category, prediction confidence, and personalized recommendations.")
    with col3:
        st.warning("**📊 Model Info**\n\nPowered by Random Forest trained on 30,000 UCI credit card records.")

    st.markdown("---")
    st.markdown("### 📖 About This System")
    st.markdown("""
    This system uses **machine learning** to predict the likelihood of a credit card customer 
    defaulting on payment next month. It analyzes:
    - **Demographic factors** – Age, education, marital status
    - **Credit behavior** – Credit limit utilization, billing patterns  
    - **Payment history** – Past payment delays (strongest predictor)
    
    The model outputs a **risk score (0–100)** categorized into:
    - 🟢 **Low Risk** (0–30): Customer is likely to pay on time
    - 🟡 **Medium Risk** (31–70): Monitor closely, consider intervention
    - 🔴 **High Risk** (71–100): Immediate action recommended
    """)

else:
    if model is None:
        st.error("⚠️ Model files not found. Please run `python main_analysis.py` first to train and save the models.")
        st.stop()

    # Build input vector in correct column order
    input_dict = {
        "LIMIT_BAL": limit_bal, "SEX": sex_val, "EDUCATION": edu_val,
        "MARRIAGE": mar_val, "AGE": age,
        "PAY_1": pay1, "PAY_2": pay2, "PAY_3": pay3,
        "PAY_4": pay4, "PAY_5": pay5, "PAY_6": pay6,
        "BILL_AMT1": bill1, "BILL_AMT2": bill2, "BILL_AMT3": bill3,
        "BILL_AMT4": bill4, "BILL_AMT5": bill5, "BILL_AMT6": bill6,
        "PAY_AMT1": pay_amt1, "PAY_AMT2": pay_amt2, "PAY_AMT3": pay_amt3,
        "PAY_AMT4": pay_amt4, "PAY_AMT5": pay_amt5, "PAY_AMT6": pay_amt6,
    }
    input_df = pd.DataFrame([input_dict])
    # Align columns
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)

    prob        = model.predict_proba(input_scaled)[0][1]
    prediction  = int(prob >= 0.5)
    score       = compute_risk_score(prob)
    level, color, emoji = get_risk_level(score)
    recs        = get_recommendations(score, limit_bal, pay1, age)

    # ── Result Header ──
    if prediction == 1:
        st.error(f"## {emoji} Prediction: **LIKELY TO DEFAULT** (Confidence: {prob*100:.1f}%)")
    else:
        st.success(f"## {emoji} Prediction: **NOT LIKELY TO DEFAULT** (Confidence: {(1-prob)*100:.1f}%)")

    # ── Key Metrics Row ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Risk Score", f"{score}/100")
    with c2:
        st.metric("Risk Level", level)
    with c3:
        st.metric("Default Probability", f"{prob*100:.1f}%")
    with c4:
        st.metric("Safe Probability", f"{(1-prob)*100:.1f}%")

    st.markdown("---")

    # ── Gauge + Feature Analysis ──
    col_gauge, col_analysis = st.columns([1, 1.5])

    with col_gauge:
        st.markdown("### 🎯 Risk Gauge")
        st.plotly_chart(make_gauge(score, color), use_container_width=True)

        risk_df = pd.DataFrame({
            "Category": ["Low Risk (0–30)", "Medium Risk (31–70)", "High Risk (71–100)"],
            "Range":    [30, 40, 30],
        })
        fig_donut = go.Figure(go.Pie(
            labels=risk_df["Category"], values=risk_df["Range"],
            hole=0.55, marker_colors=["#c6f6d5", "#fef3c7", "#fed7d7"],
            textinfo="label+percent"
        ))
        fig_donut.update_layout(height=220, showlegend=False,
                                margin=dict(t=10, b=10, l=10, r=10),
                                paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_analysis:
        st.markdown("### 📊 Input Feature Summary")
        features_display = {
            "Credit Limit":       f"NTD {limit_bal:,}",
            "Age":                str(age),
            "Payment Delay M1":   f"{pay1} month(s)",
            "Bill Amount M1":     f"NTD {bill1:,}",
            "Payment Amount M1":  f"NTD {pay_amt1:,}",
            "Utilization Ratio":  f"{min(bill1/limit_bal*100, 100):.1f}%" if limit_bal > 0 else "N/A",
        }
        for feat, val in features_display.items():
            cols = st.columns([2, 1])
            cols[0].write(feat)
            cols[1].write(f"**{val}**")

        # Payment history chart
        months = [f"M{i}" for i in range(1, 7)]
        pay_delays = [pay1, pay2, pay3, pay4, pay5, pay6]
        fig_pay = go.Figure(go.Bar(
            x=months, y=pay_delays,
            marker_color=["#fc8181" if p > 0 else "#68d391" for p in pay_delays],
            text=pay_delays, textposition="outside"
        ))
        fig_pay.update_layout(
            title="Payment Delay History (months)", height=250,
            yaxis_title="Delay (months)", xaxis_title="Month",
            margin=dict(t=40, b=20, l=30, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_pay, use_container_width=True)

    st.markdown("---")

    # ── Recommendations ──
    st.markdown("### 💡 Personalized Action Recommendations")
    for rec in recs:
        st.markdown(f"""<div class="rec-box">{rec}</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Customer Summary Card ──
    with st.expander("📋 View Full Customer Profile"):
        st.dataframe(
            pd.DataFrame([input_dict]).T.rename(columns={0: "Value"}),
            use_container_width=True
        )

    st.caption("⚠️ This tool is for academic demonstration. Consult a certified financial expert for actual credit decisions.")

# ─── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Credit Card Default Analysis System · B.Tech Data Science Project · Built with Streamlit & Scikit-learn</small></center>",
    unsafe_allow_html=True
)
