import sys
import os

# Ensure project root directory is in sys.path for Streamlit
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_preprocessing import create_features
try:
    from dashboard.pdf_generator import generate_pdf_bytes
except ImportError:
    from pdf_generator import generate_pdf_bytes

st.set_page_config(
    page_title="Medical Cost Predictor & Risk Assessment",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Medical Insurance Cost Predictor & MLOps System")
st.markdown("*Predict annual medical insurance expenditures with confidence intervals and interactive What-If scenario simulator.*")

@st.cache_resource
def load_app_models():
    model_path = "models/best_model.pkl"
    prep_path = "models/preprocessor.pkl"
    q_low_path = "models/quantile_low.pkl"
    q_high_path = "models/quantile_high.pkl"

    if not (os.path.exists(model_path) and os.path.exists(prep_path)):
        return None, None, None, None

    model = joblib.load(model_path)
    preprocessor = joblib.load(prep_path)
    q_low = joblib.load(q_low_path) if os.path.exists(q_low_path) else None
    q_high = joblib.load(q_high_path) if os.path.exists(q_high_path) else None
    return model, preprocessor, q_low, q_high

model, preprocessor, q_low, q_high = load_app_models()

if model is None:
    st.error("⚠️ Model artifacts not found. Please train models first via `python -m src.model_training` or the notebook!")
    st.stop()

# Sidebar - Patient Input Form
st.sidebar.header("📋 Patient Profile Inputs")
age = st.sidebar.slider("Age", 18, 80, 35)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.slider("Body Mass Index (BMI)", 15.0, 50.0, 27.5, step=0.1)
children = st.sidebar.slider("Number of Children", 0, 5, 1)
smoker = st.sidebar.radio("Smoking Status", ["no", "yes"])
region = st.sidebar.selectbox("US Region", ["southwest", "southeast", "northwest", "northeast"])

# Prepare Input Data
input_data = {
    "age": age, "sex": sex, "bmi": bmi,
    "children": children, "smoker": smoker, "region": region
}
input_df = pd.DataFrame([input_data])
engineered_df = create_features(input_df)

processed = preprocessor.transform(engineered_df)
pred_cost = float(np.expm1(model.predict(processed)[0]))

if q_low and q_high:
    low_bound = float(np.expm1(q_low.predict(processed)[0]))
    high_bound = float(np.expm1(q_high.predict(processed)[0]))
else:
    low_bound = pred_cost * 0.85
    high_bound = pred_cost * 1.15

# Main UI Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediction & Range", "⚡ What-If Simulator", "📊 Analytics & PDF Export"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("10th Percentile (Low)", f"${low_bound:,.2f}")
    col2.metric("Median Cost Prediction", f"${pred_cost:,.2f}", delta=None)
    col3.metric("90th Percentile (High)", f"${high_bound:,.2f}")

    st.subheader("Patient Parameter Summary")
    st.json(input_data)

with tab2:
    st.subheader("⚡ 'What-If' Scenario Simulator")
    st.write("Toggle lifestyles to see instant impact on medical insurance charges:")

    if smoker == "yes":
        toggle_smoker = "no"
        sim_df = input_df.copy()
        sim_df["smoker"] = toggle_smoker
        sim_engineered = create_features(sim_df)
        sim_proc = preprocessor.transform(sim_engineered)
        sim_cost = float(np.expm1(model.predict(sim_proc)[0]))
        savings = pred_cost - sim_cost
        st.success(f"💡 If patient **quits smoking**, predicted annual cost drops to **${sim_cost:,.2f}** (Savings: **${savings:,.2f}/yr**)!")

    sim_bmi = st.slider("Simulated Target BMI", 18.5, 40.0, float(bmi), key="sim_bmi")
    sim_df2 = input_df.copy()
    sim_df2["bmi"] = sim_bmi
    sim_engineered2 = create_features(sim_df2)
    sim_proc2 = preprocessor.transform(sim_engineered2)
    sim_cost2 = float(np.expm1(model.predict(sim_proc2)[0]))
    diff = sim_cost2 - pred_cost
    st.info(f"Modifying BMI to **{sim_bmi}** shifts cost to **${sim_cost2:,.2f}** (Difference: **${diff:+,.2f}**).")

with tab3:
    st.subheader("📄 Download Formal Patient PDF Report")
    pdf_data = bytes(generate_pdf_bytes(input_data, pred_cost, low_bound, high_bound))
    st.download_button(
        label="📥 Download PDF Patient Report",
        data=pdf_data,
        file_name=f"Medical_Risk_Report_Age{age}_{smoker}.pdf",
        mime="application/pdf"
    )
