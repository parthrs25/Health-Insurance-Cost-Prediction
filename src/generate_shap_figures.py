"""Generate SHAP figures for the final report."""
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src is importable
sys.path.insert(0, os.path.abspath("."))

from src.data_preprocessing import create_features, build_preprocessor

# Load data and apply feature engineering
df = pd.read_csv("data/raw/insurance.csv")
df = create_features(df)

# Separate features and target
target = "charges"
feature_cols = [c for c in df.columns if c != target]
X = df[feature_cols]
y = np.log1p(df[target])

# Load preprocessor (saved with joblib) and model
preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/best_model.pkl")

# Transform features
X_transformed = preprocessor.transform(X)

# Get feature names from preprocessor
try:
    feature_names = list(preprocessor.get_feature_names_out())
except Exception:
    feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

# Clean up feature names for display
feature_names = [n.replace("num__", "").replace("cat__", "") for n in feature_names]

print(f"Data shape: {X_transformed.shape}")
print(f"Feature names: {feature_names}")

# SHAP
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_transformed)

# Create figures directory
os.makedirs("reports/figures", exist_ok=True)

# 1. Bar summary plot
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names,
                  plot_type="bar", show=False, max_display=15)
plt.title("SHAP Global Feature Importance", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("reports/figures/shap_summary_bar.png", dpi=150, bbox_inches="tight")
plt.close("all")
print("Saved: shap_summary_bar.png")

# 2. Beeswarm / dot plot
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names,
                  show=False, max_display=15)
plt.title("SHAP Feature Value Impact", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("reports/figures/shap_summary_dot.png", dpi=150, bbox_inches="tight")
plt.close("all")
print("Saved: shap_summary_dot.png")

# 3. Waterfall for patient 0
plt.figure(figsize=(10, 7))
explanation = shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    feature_names=feature_names
)
shap.plots.waterfall(explanation, show=False, max_display=12)
plt.title("Patient #0 Prediction Attribution", fontsize=12, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig("reports/figures/shap_waterfall_patient_0.png", dpi=150, bbox_inches="tight")
plt.close("all")
print("Saved: shap_waterfall_patient_0.png")

print("\nAll SHAP figures generated successfully!")
